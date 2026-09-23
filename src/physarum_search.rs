//! Feature-gated Physarum-style branch-flow search.
//!
//! Conductivity chooses where fixed search traffic goes. It never supplies a
//! chess score and never chooses the played move; backed-up root-relative
//! values do that. The pre-RL implementation uses a uniform policy prior so
//! the fixed dynamics can be tested without a learned component.

use std::{sync::Arc, sync::atomic::Ordering, time::Instant};

use crate::{
    board::{Board, NullBoardObserver},
    branch_flow::{Network, NodeId, Parameters},
    conductivity_head::Head,
    search::quiescence_evaluate,
    thread::{SharedContext, Status, ThreadData},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    types::{Color, MAX_PLY, Move, normalize_to_cp},
};

// Like native Reckless, depth is an emergent consequence of the active UCI
// limit.  The option remains available for controlled experiments, but the
// normal setting must not impose an artificial depth-4 ceiling.
const DEFAULT_MAX_DEPTH: usize = MAX_PLY;
const DEFAULT_BATCH_SIZE: usize = 64;
const INFO_INTERVAL_MS: u128 = 250;
const TERMINAL_SCORE: i32 = 30_000;

#[derive(Clone)]
pub struct Runtime {
    pub maximum_depth: usize,
    pub batch_size: usize,
    pub budget: usize,
    pub qnodes: u64,
    pub seed: u64,
    pub learned: bool,
    head: Arc<Head>,
    pub diagnostic_prior_move: Option<String>,
    pub diagnostic_prior_mass_permille: usize,
    flow: Parameters,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            maximum_depth: DEFAULT_MAX_DEPTH,
            batch_size: DEFAULT_BATCH_SIZE,
            budget: 4096,
            qnodes: 4096,
            seed: 2026,
            learned: true,
            head: Arc::new(
                Head::from_bytes(include_bytes!(
                    "../experiments/computation_allocation/artifacts/conductivity-u29.bin"
                ))
                .expect("embedded conductivity model must be valid"),
            ),
            diagnostic_prior_move: None,
            diagnostic_prior_mass_permille: 990,
            flow: Parameters::default(),
        }
    }
}

impl Runtime {
    pub fn load_weights(&mut self, path: &str) -> Result<u32, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("failed to read conductivity weights: {error}"))?;
        let head = Head::from_bytes(&bytes)?;
        let update = head.update;
        self.head = Arc::new(head);
        Ok(update)
    }

    pub fn set_budget(&mut self, value: &str) {
        self.budget = value.parse().unwrap_or(4096).clamp(1, 1_000_000);
    }

    pub fn set_qnodes(&mut self, value: &str) {
        self.qnodes = value.parse().unwrap_or(4096).clamp(1, 1_000_000);
    }

    pub fn set_seed(&mut self, value: &str) {
        self.seed = value.parse().unwrap_or(2026);
    }

    pub fn set_learned(&mut self, value: &str) {
        self.learned = value.parse().unwrap_or(true);
    }

    pub fn set_maximum_depth(&mut self, value: &str) {
        self.maximum_depth = value.parse().unwrap_or(DEFAULT_MAX_DEPTH).clamp(1, MAX_PLY);
    }

    pub fn set_batch_size(&mut self, value: &str) {
        self.batch_size = value.parse().unwrap_or(DEFAULT_BATCH_SIZE).clamp(1, 64);
    }

    pub fn set_diagnostic_prior_move(&mut self, value: &str) {
        self.diagnostic_prior_move = (value != "none").then(|| value.to_string());
    }

    pub fn set_diagnostic_prior_mass(&mut self, value: &str) {
        self.diagnostic_prior_mass_permille = value.parse().unwrap_or(990).clamp(1, 999);
    }

    fn root_priors(&self, board: &Board, moves: &[Move]) -> Vec<f64> {
        let Some(target) = &self.diagnostic_prior_move else {
            return uniform_priors(moves.len());
        };
        let Some(index) = moves.iter().position(|mv| mv.to_uci(board) == *target) else {
            return uniform_priors(moves.len());
        };
        if moves.len() == 1 {
            return vec![1.0];
        }
        let target_mass = self.diagnostic_prior_mass_permille as f64 / 1_000.0;
        let remainder = (1.0 - target_mass) / (moves.len() - 1) as f64;
        let mut priors = vec![remainder; moves.len()];
        priors[index] = target_mass;
        priors
    }
}

struct PositionNode {
    board: Option<Board>,
    incoming_move: Option<Move>,
    depth: usize,
    evaluation: i32,
    value: i32,
    completed_depth: usize,
    evaluated: bool,
    terminal: bool,
}

impl PositionNode {
    fn board(&self) -> &Board {
        self.board.as_ref().expect("evaluated Physarum node must have a board")
    }
}

struct Expansion {
    node: NodeId,
    moves: Vec<Move>,
    surprise: f64,
    terminal: bool,
}

struct SearchResult {
    best_move: Move,
    score: i32,
    partial_best_move: Move,
    partial_score: i32,
    depth: usize,
    nodes: u64,
    rounds: u64,
    root_children: Vec<(Move, i32, bool)>,
}

fn root_relative_eval(td: &mut ThreadData, board: &Board, root_color: Color) -> i32 {
    let score = quiescence_evaluate(td, board);
    if board.side_to_move() == root_color { score } else { -score }
}

fn uniform_priors(count: usize) -> Vec<f64> {
    vec![1.0 / count as f64; count]
}

fn add_children(
    flow: &mut Network, positions: &mut Vec<PositionNode>, parent: NodeId, moves: Vec<Move>, priors: Vec<f64>,
) {
    debug_assert_eq!(flow.node_count(), positions.len());
    let children = flow.expand(parent, &priors, &vec![1.0; moves.len()]);
    let parent_value = positions[parent].value;
    let depth = positions[parent].depth + 1;
    for (child, mv) in children.into_iter().zip(moves) {
        debug_assert_eq!(child, positions.len());
        positions.push(PositionNode {
            board: None,
            incoming_move: Some(mv),
            depth,
            evaluation: parent_value,
            value: parent_value,
            completed_depth: 0,
            evaluated: false,
            terminal: false,
        });
    }
}

fn legal_moves(board: &Board) -> Vec<Move> {
    board.generate_all_moves().iter().map(|entry| entry.mv).collect()
}

fn materialize_board(flow: &Network, positions: &mut [PositionNode], node: NodeId) {
    if positions[node].board.is_some() {
        return;
    }
    let parent = flow.parent(node).expect("only the root lacks a parent");
    let mut board = positions[parent].board().clone();
    board.make_move(positions[node].incoming_move.unwrap(), &mut NullBoardObserver);
    positions[node].board = Some(board);
}

fn recompute_node(flow: &Network, positions: &mut [PositionNode], node: NodeId, root_color: Color) {
    if !positions[node].evaluated {
        return;
    }
    if positions[node].terminal {
        positions[node].completed_depth = MAX_PLY;
        return;
    }
    let children = flow.children(node).collect::<Vec<_>>();
    if children.is_empty() {
        positions[node].completed_depth = 0;
        return;
    }
    let maximizing = positions[node].board().side_to_move() == root_color;
    positions[node].value = if maximizing {
        children.iter().map(|child| positions[*child].value).max().unwrap()
    } else {
        children.iter().map(|child| positions[*child].value).min().unwrap()
    };
    positions[node].completed_depth = if children.iter().all(|child| positions[*child].evaluated) {
        1 + children.iter().map(|child| positions[*child].completed_depth).min().unwrap()
    } else {
        0
    };
}

fn backup(flow: &Network, positions: &mut [PositionNode], start: NodeId, root_color: Color) {
    let mut current = Some(start);
    while let Some(node) = current {
        recompute_node(flow, positions, node, root_color);
        current = flow.parent(node);
    }
}

fn preferred_child(flow: &Network, positions: &[PositionNode], node: NodeId, root_color: Color) -> Option<NodeId> {
    let maximizing = positions[node].board().side_to_move() == root_color;
    flow.children(node).max_by(|left, right| {
        let ordering = if maximizing {
            positions[*left].value.cmp(&positions[*right].value)
        } else {
            positions[*right].value.cmp(&positions[*left].value)
        };
        ordering
            .then_with(|| positions[*left].evaluated.cmp(&positions[*right].evaluated))
            .then_with(|| {
                flow.conductivity_to(*left)
                    .unwrap_or_default()
                    .total_cmp(&flow.conductivity_to(*right).unwrap_or_default())
            })
            .then_with(|| right.cmp(left))
    })
}

fn partial_best_root_child(flow: &Network, positions: &[PositionNode], root_color: Color) -> NodeId {
    let evaluated = flow.children(0).filter(|child| positions[*child].evaluated).collect::<Vec<_>>();
    if evaluated.is_empty() {
        preferred_child(flow, positions, 0, root_color).unwrap()
    } else {
        evaluated
            .into_iter()
            .max_by(|left, right| {
                positions[*left]
                    .value
                    .cmp(&positions[*right].value)
                    .then_with(|| {
                        flow.conductivity_to(*left)
                            .unwrap_or_default()
                            .total_cmp(&flow.conductivity_to(*right).unwrap_or_default())
                    })
                    .then_with(|| right.cmp(left))
            })
            .unwrap()
    }
}

fn value_at_depth(flow: &Network, positions: &[PositionNode], node: NodeId, depth: usize, root_color: Color) -> i32 {
    if depth == 0 || positions[node].terminal {
        return positions[node].evaluation;
    }
    let maximizing = positions[node].board().side_to_move() == root_color;
    let values = flow.children(node).map(|child| value_at_depth(flow, positions, child, depth - 1, root_color));
    if maximizing { values.max().unwrap() } else { values.min().unwrap() }
}

fn published_root(flow: &Network, positions: &[PositionNode], root_color: Color) -> (NodeId, i32, usize) {
    let depth = positions[0].completed_depth;
    if depth == 0 {
        let best = partial_best_root_child(flow, positions, root_color);
        return (best, positions[best].value, 0);
    }
    let child_depth = depth - 1;
    let best = flow
        .children(0)
        .max_by(|left, right| {
            value_at_depth(flow, positions, *left, child_depth, root_color)
                .cmp(&value_at_depth(flow, positions, *right, child_depth, root_color))
                .then_with(|| right.cmp(left))
        })
        .unwrap();
    (best, value_at_depth(flow, positions, best, child_depth, root_color), depth)
}

fn principal_variation(
    flow: &Network, positions: &[PositionNode], root_color: Color, mut remaining_depth: usize,
) -> Vec<String> {
    let mut pv = Vec::new();
    let mut node = 0;
    let mut next = Some(published_root(flow, positions, root_color).0);
    while let Some(child) = next {
        if let Some(mv) = positions[child].incoming_move {
            pv.push(mv.to_uci(positions[node].board()));
        }
        if !positions[child].evaluated || remaining_depth <= 1 {
            break;
        }
        remaining_depth -= 1;
        node = child;
        let maximizing = positions[node].board().side_to_move() == root_color;
        next = flow.children(node).max_by(|left, right| {
            let ordering = if maximizing {
                value_at_depth(flow, positions, *left, remaining_depth - 1, root_color).cmp(&value_at_depth(
                    flow,
                    positions,
                    *right,
                    remaining_depth - 1,
                    root_color,
                ))
            } else {
                value_at_depth(flow, positions, *right, remaining_depth - 1, root_color).cmp(&value_at_depth(
                    flow,
                    positions,
                    *left,
                    remaining_depth - 1,
                    root_color,
                ))
            };
            ordering.then_with(|| right.cmp(left))
        });
    }
    pv
}

fn print_info(
    flow: &Network, positions: &[PositionNode], root_color: Color, nodes: u64, rounds: u64, elapsed_ms: u128,
) {
    let (_, score, depth) = published_root(flow, positions, root_color);
    let display_score = normalize_to_cp(score, positions[0].board());
    let nps = nodes as u128 * 1_000 / elapsed_ms.max(1);
    print!(
        "info depth {depth} seldepth {depth} multipv 1 score cp {} nodes {nodes} time {elapsed_ms} nps {nps} string physarum-rounds {rounds} pv",
        display_score
    );
    for mv in principal_variation(flow, positions, root_color, depth) {
        print!(" {mv}");
    }
    println!();
}

fn fixed_usefulness(surprise: f64, terminal: bool, root_effect: f64, decision_changed: bool) -> f64 {
    (0.05
        + 0.35 * surprise.min(1.0)
        + 0.35 * root_effect.min(1.0)
        + 0.25 * f64::from(decision_changed)
        + 0.5 * f64::from(terminal))
    .min(1.0)
}

fn run_search(
    runtime: &Runtime, threads: &mut ThreadPool, board: &Board, shared: &Arc<SharedContext>, limits: Limits,
    move_overhead: u64, report: bool,
) -> Option<SearchResult> {
    let moves = legal_moves(board);
    if moves.is_empty() {
        return None;
    }

    let root_color = board.side_to_move();
    let requested_depth = match &limits {
        Limits::Depth(depth) => Some((*depth).max(1) as usize),
        _ => None,
    };
    let maximum_depth = requested_depth.unwrap_or(runtime.maximum_depth).min(runtime.maximum_depth);
    let manager = TimeManager::new(limits, board.fullmove_number(), move_overhead);
    shared.nodes.reset();
    shared.status.set(Status::RUNNING);
    let root_value = root_relative_eval(threads.main_thread(), board, root_color);
    let root_nodes = 1 + shared.nodes.aggregate();
    let mut positions = vec![PositionNode {
        board: Some(board.clone()),
        incoming_move: None,
        depth: 0,
        evaluation: root_value,
        value: root_value,
        completed_depth: 0,
        evaluated: true,
        terminal: false,
    }];
    let mut flow = Network::new(runtime.flow);
    let root_priors = runtime.root_priors(board, &moves);
    add_children(&mut flow, &mut positions, 0, moves, root_priors);

    let mut nodes = root_nodes;
    let mut rounds = 0_u64;
    let mut last_info = Instant::now();
    let mut previous_best = partial_best_root_child(&flow, &positions, root_color);
    if report {
        let prior = runtime.diagnostic_prior_move.as_deref().unwrap_or("uniform");
        println!("info string Physarum search enabled prior {prior} flow fixed-current batch {}", runtime.batch_size);
    }

    loop {
        if requested_depth.is_some_and(|_| positions[0].completed_depth >= maximum_depth)
            || manager.hard_limit_reached(nodes)
            || shared.externally_stopped.load(Ordering::Acquire)
        {
            break;
        }
        let batch = flow.select_batch(1.0, runtime.batch_size);
        if batch.is_empty() {
            break;
        }

        let old_root_value = positions[0].value;
        let old_best = partial_best_root_child(&flow, &positions, root_color);
        let mut expansions = Vec::with_capacity(batch.len());
        for node in &batch {
            if shared.externally_stopped.load(Ordering::Acquire) {
                break;
            }
            let old_value = positions[*node].value;
            let depth = positions[*node].depth;
            materialize_board(&flow, &mut positions, *node);
            let qnodes_before = shared.nodes.aggregate();
            let value = if positions[*node].board().is_draw(depth as isize) {
                0
            } else {
                root_relative_eval(threads.main_thread(), positions[*node].board(), root_color)
            };
            if shared.externally_stopped.load(Ordering::Acquire) {
                break;
            }
            positions[*node].evaluation = value;
            positions[*node].value = value;
            positions[*node].evaluated = true;
            nodes += 1 + shared.nodes.aggregate().saturating_sub(qnodes_before);

            let moves = legal_moves(positions[*node].board());
            let terminal = moves.is_empty() || positions[*node].board().is_draw(depth as isize);
            if moves.is_empty() {
                positions[*node].terminal = true;
                positions[*node].value = if positions[*node].board().in_check() {
                    if positions[*node].board().side_to_move() == root_color {
                        -TERMINAL_SCORE + depth as i32
                    } else {
                        TERMINAL_SCORE - depth as i32
                    }
                } else {
                    0
                };
                positions[*node].evaluation = positions[*node].value;
            } else if terminal {
                positions[*node].terminal = true;
                positions[*node].value = 0;
                positions[*node].evaluation = 0;
            }
            let surprise = f64::from((positions[*node].value - old_value).abs()) / 400.0;
            expansions.push(Expansion { node: *node, moves, surprise: surprise.min(1.0), terminal });
        }

        for expansion in &expansions {
            backup(&flow, &mut positions, expansion.node, root_color);
        }
        let new_best = partial_best_root_child(&flow, &positions, root_color);
        let root_effect = f64::from((positions[0].value - old_root_value).abs()) / 400.0;
        let decision_changed = new_best != old_best;
        let events = expansions
            .iter()
            .map(|expansion| {
                let decision_relevant =
                    matches!(flow.root_child(expansion.node), root if root == old_best || root == new_best);
                (
                    expansion.node,
                    fixed_usefulness(
                        expansion.surprise,
                        expansion.terminal,
                        if decision_relevant { root_effect } else { 0.0 },
                        decision_changed && decision_relevant,
                    ),
                )
            })
            .collect::<Vec<_>>();
        flow.adapt(&events);

        for expansion in expansions {
            if expansion.terminal || positions[expansion.node].depth >= maximum_depth {
                flow.close_frontier(expansion.node);
            } else {
                let priors = uniform_priors(expansion.moves.len());
                add_children(&mut flow, &mut positions, expansion.node, expansion.moves, priors);
            }
        }
        backup(&flow, &mut positions, 0, root_color);
        rounds += 1;

        let current_best = published_root(&flow, &positions, root_color).0;
        if report && (current_best != previous_best || last_info.elapsed().as_millis() >= INFO_INTERVAL_MS) {
            print_info(&flow, &positions, root_color, nodes, rounds, manager.elapsed().as_millis());
            previous_best = current_best;
            last_info = Instant::now();
        }
    }

    let (best, score, depth) = published_root(&flow, &positions, root_color);
    let partial_best = partial_best_root_child(&flow, &positions, root_color);
    if report {
        print_info(&flow, &positions, root_color, nodes, rounds, manager.elapsed().as_millis());
    }
    Some(SearchResult {
        best_move: positions[best].incoming_move.unwrap(),
        score,
        partial_best_move: positions[partial_best].incoming_move.unwrap(),
        partial_score: positions[partial_best].value,
        depth,
        nodes,
        rounds,
        root_children: flow
            .children(0)
            .map(|child| (positions[child].incoming_move.unwrap(), positions[child].value, positions[child].evaluated))
            .collect(),
    })
}

pub fn go(
    runtime: &Runtime, threads: &mut ThreadPool, board: &Board, shared: &Arc<SharedContext>, limits: Limits,
    move_overhead: u64,
) {
    crate::learned_physarum_search::go(
        &runtime.head,
        runtime.learned,
        runtime.budget,
        runtime.maximum_depth,
        runtime.qnodes,
        runtime.seed,
        threads,
        board,
        shared,
        limits,
        move_overhead,
    );
}
