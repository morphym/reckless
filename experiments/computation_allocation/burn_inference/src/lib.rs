use std::path::Path;

use burn::{
    backend::Flex,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Device, Tensor, TensorData, activation::silu, backend::Backend},
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

pub const CANDIDATE_FEATURES: usize = 18;
pub const GLOBAL_FEATURES: usize = 12;
const CANDIDATE_WIDTH: usize = 128;
const GLOBAL_WIDTH: usize = 96;
const CONTEXT_WIDTH: usize = 192;
const HIDDEN_WIDTH: usize = 128;

type CpuBackend = Flex<f32>;

#[derive(Module, Debug)]
struct CsActor<B: Backend> {
    candidate_linear_1: Linear<B>,
    candidate_norm: LayerNorm<B>,
    candidate_linear_2: Linear<B>,
    global_linear: Linear<B>,
    global_norm: LayerNorm<B>,
    context_linear: Linear<B>,
    context_norm: LayerNorm<B>,
    candidate_actor_hidden: Linear<B>,
    candidate_actor_output: Linear<B>,
    stop_actor_hidden: Linear<B>,
    stop_actor_output: Linear<B>,
}

impl<B: Backend> CsActor<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            candidate_linear_1: LinearConfig::new(CANDIDATE_FEATURES, CANDIDATE_WIDTH).init(device),
            candidate_norm: LayerNormConfig::new(CANDIDATE_WIDTH).init(device),
            candidate_linear_2: LinearConfig::new(CANDIDATE_WIDTH, CANDIDATE_WIDTH).init(device),
            global_linear: LinearConfig::new(GLOBAL_FEATURES, GLOBAL_WIDTH).init(device),
            global_norm: LayerNormConfig::new(GLOBAL_WIDTH).init(device),
            context_linear: LinearConfig::new(2 * CANDIDATE_WIDTH + GLOBAL_WIDTH, CONTEXT_WIDTH).init(device),
            context_norm: LayerNormConfig::new(CONTEXT_WIDTH).init(device),
            candidate_actor_hidden: LinearConfig::new(CANDIDATE_WIDTH + CONTEXT_WIDTH, HIDDEN_WIDTH).init(device),
            candidate_actor_output: LinearConfig::new(HIDDEN_WIDTH, 1).init(device),
            stop_actor_hidden: LinearConfig::new(CONTEXT_WIDTH, HIDDEN_WIDTH).init(device),
            stop_actor_output: LinearConfig::new(HIDDEN_WIDTH, 1).init(device),
        }
    }

    /// Encode every branch together and reuse one pooled context for all action logits.
    fn forward(&self, candidates: Tensor<B, 2>, globals: Tensor<B, 2>) -> Tensor<B, 1> {
        let count = candidates.dims()[0];
        let encoded = self.candidate_linear_1.forward(candidates);
        let encoded = self.candidate_norm.forward(silu(encoded));
        let encoded = silu(self.candidate_linear_2.forward(encoded));
        let mean = encoded.clone().sum_dim(0) / count as f32;
        let maximum = encoded.clone().max_dim(0);
        let globals = self.global_norm.forward(silu(self.global_linear.forward(globals)));
        let context = Tensor::cat(vec![mean, maximum, globals], 1);
        let context = self.context_norm.forward(silu(self.context_linear.forward(context)));

        let repeated = context.clone().repeat_dim(0, count);
        let candidate_context = Tensor::cat(vec![encoded, repeated], 1);
        let candidate_hidden = silu(self.candidate_actor_hidden.forward(candidate_context));
        let candidate_logits = self.candidate_actor_output.forward(candidate_hidden).reshape([count]);
        let stop_hidden = silu(self.stop_actor_hidden.forward(context));
        let stop_logit = self.stop_actor_output.forward(stop_hidden).reshape([1]);
        Tensor::cat(vec![candidate_logits, stop_logit], 0)
    }
}

/// Resident, inference-only CS actor. Model construction and weight loading are amortized.
pub struct Inference {
    model: CsActor<CpuBackend>,
    device: Device<CpuBackend>,
}

impl Inference {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Default::default();
        let mut model = CsActor::<CpuBackend>::init(&device);
        let mut store = SafetensorsStore::from_file(path).with_from_adapter(PyTorchToBurnAdapter);
        model.load_from(&mut store)?;
        Ok(Self { model, device })
    }

    pub fn logits(&self, candidates: &[f32], globals: &[f32], count: usize) -> Vec<f32> {
        assert!(count > 0);
        assert_eq!(candidates.len(), count * CANDIDATE_FEATURES);
        assert_eq!(globals.len(), GLOBAL_FEATURES);
        let candidates = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(candidates.to_vec(), [count, CANDIDATE_FEATURES]),
            &self.device,
        );
        let globals =
            Tensor::<CpuBackend, 2>::from_data(TensorData::new(globals.to_vec(), [1, GLOBAL_FEATURES]), &self.device);
        self.model.forward(candidates, globals).into_data().to_vec::<f32>().expect("Burn output should be f32")
    }

    pub fn masked_logits(&self, candidates: &[f32], globals: &[f32], action_mask: &[bool], count: usize) -> Vec<f32> {
        assert_eq!(action_mask.len(), count + 1);
        let mut logits = self.logits(candidates, globals, count);
        for (logit, legal) in logits.iter_mut().zip(action_mask) {
            if !legal {
                *logit = f32::MIN;
            }
        }
        logits
    }
}
