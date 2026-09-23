//! Fixed, auditable Physarum-style traffic dynamics.
//!
//! This module knows nothing about chess values. It routes a fixed source
//! current over the represented root-to-frontier tree and adapts edge
//! conductivity from useful traffic. Game-value backup belongs to the search
//! using this topology.

pub type NodeId = usize;
type EdgeId = usize;

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub exploration_floor: f64,
    pub prior_strength: f64,
    pub retention: f64,
    pub learning_rate: f64,
    pub normalize_deposit_by_cost: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            exploration_floor: 0.01,
            prior_strength: 1.0,
            retention: 0.97,
            learning_rate: 0.75,
            normalize_deposit_by_cost: true,
        }
    }
}

#[derive(Debug)]
struct FlowNode {
    parent_edge: Option<EdgeId>,
    children: Vec<EdgeId>,
    frontier: bool,
    policy_mass: f64,
    traffic_credit: f64,
    last_flow: f64,
}

#[derive(Debug)]
struct FlowEdge {
    parent: NodeId,
    child: NodeId,
    conductivity: f64,
    length: f64,
    last_flow: f64,
    pending_deposit: f64,
}

#[derive(Debug)]
pub struct Network {
    parameters: Parameters,
    nodes: Vec<FlowNode>,
    edges: Vec<FlowEdge>,
}

impl Network {
    pub fn new(parameters: Parameters) -> Self {
        assert!(parameters.exploration_floor > 0.0);
        assert!(parameters.prior_strength >= 0.0);
        assert!((0.0..1.0).contains(&parameters.retention));
        assert!(parameters.learning_rate >= 0.0);
        Self {
            parameters,
            nodes: vec![FlowNode {
                parent_edge: None,
                children: Vec::new(),
                frontier: true,
                policy_mass: 1.0,
                traffic_credit: 0.0,
                last_flow: 0.0,
            }],
            edges: Vec::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes[node].parent_edge.map(|edge| self.edge_parent(edge))
    }

    pub fn children(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes[node].children.iter().map(|edge| self.edges[*edge].child)
    }

    pub fn conductivity_to(&self, node: NodeId) -> Option<f64> {
        self.nodes[node].parent_edge.map(|edge| self.edges[edge].conductivity)
    }

    pub fn root_child(&self, mut node: NodeId) -> NodeId {
        while let Some(parent) = self.parent(node) {
            if parent == 0 {
                return node;
            }
            node = parent;
        }
        node
    }

    pub fn close_frontier(&mut self, node: NodeId) {
        self.nodes[node].frontier = false;
        self.nodes[node].traffic_credit = 0.0;
    }

    /// Replace a frontier by children initialized from prefix policy mass.
    pub fn expand(&mut self, node: NodeId, priors: &[f64], lengths: &[f64]) -> Vec<NodeId> {
        assert!(self.nodes[node].frontier);
        assert_eq!(priors.len(), lengths.len());
        assert!(!priors.is_empty());
        assert!(lengths.iter().all(|length| length.is_finite() && *length > 0.0));

        let mut cleaned =
            priors.iter().map(|prior| if prior.is_finite() { prior.max(0.0) } else { 0.0 }).collect::<Vec<_>>();
        let sum = cleaned.iter().sum::<f64>();
        if sum > 0.0 {
            for prior in &mut cleaned {
                *prior /= sum;
            }
        } else {
            let count = cleaned.len();
            cleaned.fill(1.0 / count as f64);
        }

        self.close_frontier(node);
        let prefix_mass = self.nodes[node].policy_mass;
        let mut children = Vec::with_capacity(cleaned.len());
        for (prior, length) in cleaned.into_iter().zip(lengths.iter().copied()) {
            let policy_mass = prefix_mass * prior;
            let child = self.nodes.len();
            let edge = self.edges.len();
            self.nodes.push(FlowNode {
                parent_edge: Some(edge),
                children: Vec::new(),
                frontier: true,
                policy_mass,
                traffic_credit: 0.0,
                last_flow: 0.0,
            });
            self.edges.push(FlowEdge {
                parent: node,
                child,
                conductivity: self.parameters.exploration_floor + self.parameters.prior_strength * policy_mass,
                length,
                last_flow: 0.0,
                pending_deposit: 0.0,
            });
            self.nodes[node].children.push(edge);
            children.push(child);
        }
        children
    }

    /// Solve fixed-current flow on the represented tree and return a
    /// deterministic weighted-fair batch of frontier nodes.
    pub fn select_batch(&mut self, source_flow: f64, batch_size: usize) -> Vec<NodeId> {
        if source_flow <= 0.0 || batch_size == 0 {
            return Vec::new();
        }
        for edge in &mut self.edges {
            edge.last_flow = 0.0;
        }
        for node in &mut self.nodes {
            node.last_flow = 0.0;
        }

        let mut effective = vec![0.0; self.nodes.len()];
        self.effective_conductance(0, &mut effective);
        let mut frontiers = Vec::new();
        self.route_flow(0, source_flow, &effective, &mut frontiers);
        for node in &frontiers {
            self.nodes[*node].traffic_credit += self.nodes[*node].last_flow;
        }
        let nodes = &self.nodes;
        let priority = |left: &NodeId, right: &NodeId| {
            nodes[*right]
                .traffic_credit
                .total_cmp(&nodes[*left].traffic_credit)
                .then_with(|| nodes[*right].last_flow.total_cmp(&nodes[*left].last_flow))
                .then_with(|| left.cmp(right))
        };
        if frontiers.len() > batch_size {
            frontiers.select_nth_unstable_by(batch_size, priority);
            frontiers.truncate(batch_size);
        }
        frontiers.sort_unstable_by(priority);
        frontiers
    }

    /// Apply one simultaneous conductivity update. Every event is
    /// `(frontier node, bounded usefulness)` and deposits its routed traffic
    /// along the complete root-to-frontier path.
    pub fn adapt(&mut self, events: &[(NodeId, f64)]) {
        for &(node, usefulness) in events {
            let usefulness = if usefulness.is_finite() { usefulness.max(0.0) } else { 0.0 };
            let path = self.path_edges(node);
            let path_cost = path.iter().map(|edge| self.edges[*edge].length).sum::<f64>().max(1.0);
            let normalization = if self.parameters.normalize_deposit_by_cost { path_cost } else { 1.0 };
            let deposit = self.nodes[node].last_flow.abs() * usefulness / normalization;
            for edge in path {
                self.edges[edge].pending_deposit += deposit;
            }
        }

        for edge in &mut self.edges {
            edge.conductivity = (self.parameters.retention * edge.conductivity
                + self.parameters.learning_rate * edge.pending_deposit)
                .max(self.parameters.exploration_floor);
            edge.pending_deposit = 0.0;
        }
    }

    fn edge_parent(&self, edge: EdgeId) -> NodeId {
        self.edges[edge].parent
    }

    fn path_edges(&self, mut node: NodeId) -> Vec<EdgeId> {
        let mut path = Vec::new();
        while let Some(edge) = self.nodes[node].parent_edge {
            path.push(edge);
            node = self.edge_parent(edge);
        }
        path
    }

    fn effective_conductance(&self, node: NodeId, effective: &mut [f64]) -> f64 {
        if self.nodes[node].frontier {
            effective[node] = f64::INFINITY;
            return f64::INFINITY;
        }

        let mut total = 0.0;
        for edge_id in &self.nodes[node].children {
            let edge = &self.edges[*edge_id];
            let child = self.effective_conductance(edge.child, effective);
            let edge_conductance = edge.conductivity / edge.length;
            let branch = if child.is_infinite() {
                edge_conductance
            } else if child > 0.0 {
                edge_conductance * child / (edge_conductance + child)
            } else {
                0.0
            };
            total += branch;
        }
        effective[node] = total;
        total
    }

    fn route_flow(&mut self, node: NodeId, incoming: f64, effective: &[f64], frontiers: &mut Vec<NodeId>) {
        if self.nodes[node].frontier {
            self.nodes[node].last_flow = incoming;
            frontiers.push(node);
            return;
        }

        let child_count = self.nodes[node].children.len();
        let mut total = 0.0;
        for index in 0..child_count {
            let edge = &self.edges[self.nodes[node].children[index]];
            let child = effective[edge.child];
            let edge_conductance = edge.conductivity / edge.length;
            total += if child.is_infinite() {
                edge_conductance
            } else if child > 0.0 {
                edge_conductance * child / (edge_conductance + child)
            } else {
                0.0
            };
        }
        if total <= 0.0 {
            return;
        }

        for index in 0..child_count {
            let edge_id = self.nodes[node].children[index];
            let edge = &self.edges[edge_id];
            let child_node = edge.child;
            let child = effective[child_node];
            let edge_conductance = edge.conductivity / edge.length;
            let branch = if child.is_infinite() {
                edge_conductance
            } else if child > 0.0 {
                edge_conductance * child / (edge_conductance + child)
            } else {
                0.0
            };
            if branch <= 0.0 {
                continue;
            }
            let flow = incoming * branch / total;
            self.edges[edge_id].last_flow = flow;
            self.route_flow(child_node, flow, effective, frontiers);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misleading_prior_cannot_permanently_starve_the_other_branch() {
        let mut network = Network::new(Parameters {
            retention: 0.9,
            learning_rate: 1.0,
            normalize_deposit_by_cost: false,
            ..Parameters::default()
        });
        let roots = network.expand(0, &[0.99, 0.01], &[1.0, 1.0]);

        // The misleading route receives almost all initial current, but once
        // its terminal evidence is consumed it is no longer a computation
        // sink. The exploration-floor route must then receive source current.
        assert_eq!(network.select_batch(1.0, 1), vec![roots[0]]);
        network.adapt(&[(roots[0], 0.0)]);
        network.close_frontier(roots[0]);
        assert_eq!(network.select_batch(1.0, 1), vec![roots[1]]);
    }

    #[test]
    fn useful_low_prior_traffic_reinforces_the_complete_path() {
        let mut network = Network::new(Parameters {
            retention: 0.75,
            learning_rate: 1.0,
            normalize_deposit_by_cost: false,
            ..Parameters::default()
        });
        let roots = network.expand(0, &[0.9, 0.1], &[1.0, 1.0]);
        let deep = network.expand(roots[1], &[1.0], &[1.0]);
        let before_root = network.conductivity_to(roots[1]).unwrap();
        let before_deep = network.conductivity_to(deep[0]).unwrap();

        let selected = network.select_batch(1.0, 2);
        assert!(selected.contains(&deep[0]));
        network.adapt(&[(deep[0], 4.0)]);

        assert!(network.conductivity_to(roots[1]).unwrap() > 0.75 * before_root);
        assert!(network.conductivity_to(deep[0]).unwrap() > 0.75 * before_deep);
        // The unused misleading route receives retention only.
        assert!((network.conductivity_to(roots[0]).unwrap() - 0.75 * (0.01 + 0.9)).abs() < 1e-12);
    }

    #[test]
    fn minimax_choice_is_independent_of_conductivity_choice() {
        let mut network = Network::new(Parameters {
            retention: 0.999,
            learning_rate: 0.01,
            normalize_deposit_by_cost: false,
            ..Parameters::default()
        });
        let roots = network.expand(0, &[0.999, 0.001], &[1.0, 1.0]);
        let mut values = [0, 0];

        let first = network.select_batch(1.0, 1)[0];
        assert_eq!(first, roots[0]);
        values[0] = -1_000;
        network.adapt(&[(first, 1.0)]);
        network.close_frontier(first);

        let second = network.select_batch(1.0, 1)[0];
        assert_eq!(second, roots[1]);
        values[1] = 100;
        network.adapt(&[(second, 1.0)]);

        assert!(network.conductivity_to(roots[0]).unwrap() > network.conductivity_to(roots[1]).unwrap());
        let minimax_best = values.iter().enumerate().max_by_key(|(_, value)| *value).unwrap().0;
        assert_eq!(minimax_best, 1, "the played action follows value, not the still-larger misleading conductivity");
    }
}
