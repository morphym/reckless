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
    thread::{SharedContext, Status, ThreadData},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    types::{Color, MAX_PLY, Move},
};

const DEFAULT_MAX_DEPTH: usize = 64;
const DEFAULT_BATCH_SIZE: usize = 8;
const INFO_INTERVAL_MS: u128 = 250;
const TERMINAL_SCORE: i32 = 30_000;

#[derive(Clone)]
pub struct Runtime {
    pub maximum_depth: usize,
    pub batch_size: usize,
    pub diagnostic_prior_move: Option<String>,
    pub diagnostic_prior_mass_permille: usize,
    flow: Parameters,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            maximum_depth: DEFAULT_MAX_DEPTH,
            batch_size: DEFAULT_BATCH_SIZE,
            diagnostic_prior_move: None,
            diagnostic_prior_mass_permille: 990,
            flow: Parameters::default(),
        }
    }
}

impl Runtime {
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
    board: Board,
    incoming_move: Option<Move>,
    depth: usize,
    value: i32,
    evaluated: bool,
    terminal: bool,
}

struct Expansion {
    node: NodeId,
    moves: Vec<Move>,
    boards: Vec<Board>,
    surprise: f64,
    terminal: bool,
}

struct SearchResult {
    best_move: Move,
    score: i32,
    depth: usize,
    nodes: u64,
    rounds: u64,
    root_children: Vec<(Move, i32, bool)>,
}

fn root_relative_eval(td: &mut ThreadData, board: &Board, root_color: Color) -> i32 {
    td.nnue.full_refresh(board);
    let score = td.nnue.evaluate(board);
    if board.side_to_move() == root_color { score } else { -score }
}

fn uniform_priors(count: usize) -> Vec<f64> {
    vec![1.0 / count as f64; count]
}

fn add_children(
    flow: &mut Network, positions: &mut Vec<PositionNode>, parent: NodeId, moves: Vec<Move>, boards: Vec<Board>,
    priors: Vec<f64>,
) {
    debug_assert_eq!(flow.node_count(), positions.len());
    let children = flow.expand(parent, &priors, &vec![1.0; moves.len()]);
    let parent_value = positions[parent].value;
    let depth = positions[parent].depth + 1;
    for ((child, mv), board) in children.into_iter().zip(moves).zip(boards) {
        debug_assert_eq!(child, positions.len());
        positions.push(PositionNode {
            board,
            incoming_move: Some(mv),
            depth,
            value: parent_value,
            evaluated: false,
            terminal: false,
        });
    }
}

fn make_children(board: &Board) -> (Vec<Move>, Vec<Board>) {
    let moves = board.generate_all_moves().iter().map(|entry| entry.mv).collect::<Vec<_>>();
    let boards = moves
        .iter()
        .map(|mv| {
            let mut child = board.clone();
            child.make_move(*mv, &mut NullBoardObserver);
            child
        })
        .collect();
    (moves, boards)
}

fn recompute_value(flow: &Network, positions: &mut [PositionNode], node: NodeId, root_color: Color) {
    let children = flow.children(node).collect::<Vec<_>>();
    if children.is_empty() {
        return;
    }
    let maximizing = positions[node].board.side_to_move() == root_color;
    positions[node].value = if maximizing {
        children.iter().map(|child| positions[*child].value).max().unwrap()
    } else {
        children.iter().map(|child| positions[*child].value).min().unwrap()
    };
}

fn backup(flow: &Network, positions: &mut [PositionNode], start: NodeId, root_color: Color) {
    let mut current = Some(start);
    while let Some(node) = current {
        recompute_value(flow, positions, node, root_color);
        current = flow.parent(node);
    }
}

fn preferred_child(flow: &Network, positions: &[PositionNode], node: NodeId, root_color: Color) -> Option<NodeId> {
    let maximizing = positions[node].board.side_to_move() == root_color;
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

fn best_root_child(flow: &Network, positions: &[PositionNode], root_color: Color) -> NodeId {
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

fn principal_variation(flow: &Network, positions: &[PositionNode], root_color: Color) -> Vec<String> {
    let mut pv = Vec::new();
    let mut node = 0;
    let mut next = Some(best_root_child(flow, positions, root_color));
    while let Some(child) = next {
        if let Some(mv) = positions[child].incoming_move {
            pv.push(mv.to_uci(&positions[node].board));
        }
        if !positions[child].evaluated {
            break;
        }
        node = child;
        next = preferred_child(flow, positions, node, root_color);
    }
    pv
}

fn print_info(
    flow: &Network, positions: &[PositionNode], root_color: Color, depth: usize, nodes: u64, rounds: u64,
    elapsed_ms: u128,
) {
    let best = best_root_child(flow, positions, root_color);
    let nps = nodes as u128 * 1_000 / elapsed_ms.max(1);
    print!(
        "info depth {depth} seldepth {depth} multipv 1 score cp {} nodes {nodes} time {elapsed_ms} nps {nps} string physarum-rounds {rounds} pv",
        positions[best].value
    );
    for mv in principal_variation(flow, positions, root_color) {
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
    let (moves, boards) = make_children(board);
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
    let root_value = root_relative_eval(threads.main_thread(), board, root_color);
    let mut positions = vec![PositionNode {
        board: board.clone(),
        incoming_move: None,
        depth: 0,
        value: root_value,
        evaluated: true,
        terminal: false,
    }];
    let mut flow = Network::new(runtime.flow);
    let root_priors = runtime.root_priors(board, &moves);
    add_children(&mut flow, &mut positions, 0, moves, boards, root_priors);

    let mut nodes = 0_u64;
    let mut rounds = 0_u64;
    let mut reported_depth = 0_usize;
    let mut last_info = Instant::now();
    let mut previous_best = best_root_child(&flow, &positions, root_color);
    shared.status.set(Status::RUNNING);
    if report {
        let prior = runtime.diagnostic_prior_move.as_deref().unwrap_or("uniform");
        println!("info string Physarum search enabled prior {prior} flow fixed-current batch {}", runtime.batch_size);
    }

    loop {
        if requested_depth.is_some_and(|_| reported_depth >= maximum_depth)
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
        let old_best = best_root_child(&flow, &positions, root_color);
        let mut expansions = Vec::with_capacity(batch.len());
        for node in &batch {
            if shared.externally_stopped.load(Ordering::Acquire) {
                break;
            }
            let old_value = positions[*node].value;
            let depth = positions[*node].depth;
            let value = if positions[*node].board.is_draw(depth as isize) {
                0
            } else {
                root_relative_eval(threads.main_thread(), &positions[*node].board, root_color)
            };
            positions[*node].value = value;
            positions[*node].evaluated = true;
            nodes += 1;
            reported_depth = reported_depth.max(depth);

            let (moves, boards) = make_children(&positions[*node].board);
            let terminal = moves.is_empty() || positions[*node].board.is_draw(depth as isize);
            if moves.is_empty() {
                positions[*node].terminal = true;
                positions[*node].value = if positions[*node].board.in_check() {
                    if positions[*node].board.side_to_move() == root_color {
                        -TERMINAL_SCORE + depth as i32
                    } else {
                        TERMINAL_SCORE - depth as i32
                    }
                } else {
                    0
                };
            } else if terminal {
                positions[*node].terminal = true;
                positions[*node].value = 0;
            }
            let surprise = f64::from((positions[*node].value - old_value).abs()) / 400.0;
            expansions.push(Expansion {
                node: *node,
                moves,
                boards,
                surprise: surprise.min(1.0),
                terminal,
            });
        }

        for expansion in &expansions {
            backup(&flow, &mut positions, expansion.node, root_color);
        }
        let new_best = best_root_child(&flow, &positions, root_color);
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
                add_children(&mut flow, &mut positions, expansion.node, expansion.moves, expansion.boards, priors);
            }
        }
        backup(&flow, &mut positions, 0, root_color);
        rounds += 1;

        let current_best = best_root_child(&flow, &positions, root_color);
        if report && (current_best != previous_best || last_info.elapsed().as_millis() >= INFO_INTERVAL_MS) {
            print_info(&flow, &positions, root_color, reported_depth, nodes, rounds, manager.elapsed().as_millis());
            previous_best = current_best;
            last_info = Instant::now();
        }
    }

    let best = best_root_child(&flow, &positions, root_color);
    if report {
        print_info(&flow, &positions, root_color, reported_depth, nodes, rounds, manager.elapsed().as_millis());
    }
    Some(SearchResult {
        best_move: positions[best].incoming_move.unwrap(),
        score: positions[best].value,
        depth: reported_depth,
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
    match run_search(runtime, threads, board, shared, limits, move_overhead, true) {
        Some(result) => {
            println!(
                "info string Physarum summary score {} depth {} nodes {} rounds {} root-branches {}",
                result.score,
                result.depth,
                result.nodes,
                result.rounds,
                result.root_children.len()
            );
            println!("bestmove {}", result.best_move.to_uci(board));
        }
        None => {
            println!("info depth 0 score {} 0", if board.in_check() { "mate" } else { "cp" });
            println!("bestmove (none)");
        }
    }
    shared.status.set(Status::STOPPED);
}
