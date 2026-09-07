//! Feature-gated computation-search controller for UCI `go`.
//!
//! The actor chooses which root child receives the next native Reckless search.
//! It never supplies chess values, alpha-beta bounds, or descendant moves.

use std::sync::{Arc, atomic::Ordering};

use reckless_cs_burn::Inference;

use crate::{
    board::{Board, NullBoardObserver},
    search::Report,
    thread::{SharedContext, Status},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    types::{MAX_PLY, Move, PieceType, Score, normalize_to_cp},
};

const WEIGHTS: &[u8] = include_bytes!("../experiments/computation_allocation/artifacts/controller-d8-u181.safetensors");
const CANDIDATE_FEATURES: usize = 18;
const GLOBAL_FEATURES: usize = 12;
const MATE_BASE: i32 = 100_000;
const TRAINED_CHILD_DEPTH: usize = 5;
const DEFAULT_ROOT_DEPTH: usize = 64;
const INFO_INTERVAL_MS: u128 = 250;

#[derive(Clone, Copy)]
enum BoundKind {
    Exact,
    Lower,
    Upper,
}

struct Branch {
    mv: Move,
    name: String,
    score: i32,
    mate: Option<i32>,
    depth: usize,
    seldepth: usize,
    cumulative_nodes: u64,
    last_nodes: u64,
    cumulative_time_ms: u128,
    last_score_delta: i32,
    bound: BoundKind,
    searches: usize,
    pv: Vec<Move>,
}

pub struct Runtime {
    actor: Inference,
    pub budget: usize,
    pub maximum_depth: usize,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            actor: Inference::from_bytes(WEIGHTS).expect("embedded CS actor must load"),
            budget: 32,
            maximum_depth: DEFAULT_ROOT_DEPTH,
        }
    }
}

impl Runtime {
    pub fn set_budget(&mut self, value: &str) {
        self.budget = value.parse().unwrap_or(32).clamp(1, 64);
    }

    pub fn set_maximum_depth(&mut self, value: &str) {
        self.maximum_depth = value.parse().unwrap_or(DEFAULT_ROOT_DEPTH).clamp(1, MAX_PLY);
    }
}

fn normalized_score(value: i32) -> f32 {
    (value as f64 / 1_000.0).tanh() as f32
}

fn normalized_count(value: u128, scale: f64) -> f32 {
    ((value as f64).ln_1p() / scale).min(1.0) as f32
}

fn promotion(mv: Move) -> f32 {
    if !mv.is_promotion() {
        return 0.0;
    }
    match mv.promo_piece_type() {
        PieceType::Knight => 0.25,
        PieceType::Bishop => 0.5,
        PieceType::Rook => 0.75,
        PieceType::Queen => 1.0,
        _ => 0.0,
    }
}

fn selected_index(branches: &[Branch]) -> usize {
    branches
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.score.cmp(&right.score).then_with(|| left.name.cmp(&right.name)))
        .map(|(index, _)| index)
        .unwrap()
}

fn encode(
    branches: &[Branch], remaining_budget: usize, initial_budget: usize, child_depth_limit: usize, total_nodes: u64,
    total_time_ms: u128, decision_changes: usize,
) -> (Vec<f32>, [f32; GLOBAL_FEATURES], Vec<bool>) {
    let best_score = branches.iter().map(|branch| branch.score).max().unwrap();
    let worst_score = branches.iter().map(|branch| branch.score).min().unwrap();
    let mut scores = branches.iter().map(|branch| branch.score).collect::<Vec<_>>();
    scores.sort_unstable_by(|left, right| right.cmp(left));
    let top_gap = scores.first().unwrap() - scores.get(1).unwrap_or(scores.first().unwrap());
    let selected = selected_index(branches);
    let budget_ratio = remaining_budget as f32 / initial_budget.max(1) as f32;
    // The actor was trained with child depths 1..=5. Saturating the two depth
    // features preserves that learned scale while UCI search can continue much
    // deeper using node/time/history features for additional discrimination.
    let feature_depth = TRAINED_CHILD_DEPTH;
    let mut candidates = Vec::with_capacity(branches.len() * CANDIDATE_FEATURES);
    let mut mask = Vec::with_capacity(branches.len() + 1);

    for (index, branch) in branches.iter().enumerate() {
        let legal = branch.searches < child_depth_limit && remaining_budget > 0;
        let encoded_depth = branch.depth.min(feature_depth);
        candidates.extend_from_slice(&[
            normalized_score(branch.score),
            normalized_score(branch.score - best_score),
            encoded_depth as f32 / feature_depth as f32,
            feature_depth.saturating_sub(encoded_depth) as f32 / feature_depth as f32,
            normalized_count(branch.cumulative_nodes as u128, 20.0),
            normalized_count(branch.last_nodes as u128, 16.0),
            normalized_count(branch.cumulative_time_ms, 12.0),
            normalized_score(branch.last_score_delta),
            f32::from(matches!(branch.bound, BoundKind::Exact)),
            f32::from(matches!(branch.bound, BoundKind::Lower)),
            f32::from(matches!(branch.bound, BoundKind::Upper)),
            f32::from(index == selected),
            branch.mv.from() as u8 as f32 / 63.0,
            branch.mv.to() as u8 as f32 / 63.0,
            promotion(branch.mv),
            f32::from(branch.searches > 0),
            budget_ratio,
            f32::from(legal),
        ]);
        mask.push(legal);
    }
    mask.push(true);

    let score_sum: i64 = branches.iter().map(|branch| i64::from(branch.score)).sum();
    let mean_score = (score_sum as f64 / branches.len() as f64).round_ties_even() as i32;
    let mean_depth = branches.iter().map(|branch| branch.depth).sum::<usize>() as f32 / branches.len() as f32;
    let deepest = branches.iter().map(|branch| branch.depth).max().unwrap();
    let globals = [
        budget_ratio,
        (initial_budget as f32 / 64.0).min(1.0),
        (branches.len() as f32 / 64.0).min(1.0),
        normalized_score(mean_score),
        normalized_score(best_score),
        normalized_score(worst_score),
        normalized_score(top_gap),
        (mean_depth / feature_depth as f32).min(1.0),
        (deepest as f32 / feature_depth as f32).min(1.0),
        normalized_count(total_nodes as u128, 22.0),
        normalized_count(total_time_ms, 14.0),
        (decision_changes as f32 / initial_budget.max(1) as f32).min(1.0),
    ];
    (candidates, globals, mask)
}

fn root_score(score: i32, board: &Board) -> (i32, Option<i32>) {
    match score.abs() {
        value if value < Score::TB_WIN_IN_MAX => (-normalize_to_cp(score, board), None),
        value if value <= Score::TB_WIN => {
            let cp = 20_000 - Score::TB_WIN + value;
            let child_score = if score.is_positive() { cp } else { -cp };
            (-child_score, None)
        }
        value => {
            let distance = (Score::MATE - value + i32::from(score.is_positive())) / 2;
            let child_mate = if score.is_positive() { distance } else { -distance };
            let child_score =
                if child_mate.is_positive() { MATE_BASE - child_mate * 100 } else { -MATE_BASE - child_mate * 100 };
            // Moving into a child where the opponent is mated in N means mate
            // in N+1 from this root. An opponent mate keeps its move count.
            let root_mate = if child_mate <= 0 { -child_mate + 1 } else { -child_mate };
            (-child_score, Some(root_mate))
        }
    }
}

fn static_score(threads: &mut ThreadPool, board: &Board, mv: Move) -> i32 {
    let mut child = board.clone();
    child.make_move(mv, &mut NullBoardObserver);
    let td = threads.main_thread();
    td.nnue.full_refresh(&child);
    -td.nnue.evaluate(&child)
}

struct BranchSearch {
    score: i32,
    mate: Option<i32>,
    seldepth: usize,
    nodes: u64,
    time_ms: u128,
    bound: BoundKind,
    pv: Vec<Move>,
}

fn search_branch(
    threads: &mut ThreadPool, board: &Board, shared: &Arc<SharedContext>, mv: Move, depth: usize,
) -> Option<BranchSearch> {
    let mut child = board.clone();
    child.make_move(mv, &mut NullBoardObserver);
    if child.generate_all_moves().is_empty() {
        return Some(BranchSearch {
            score: if child.in_check() { MATE_BASE } else { 0 },
            mate: child.in_check().then_some(1),
            seldepth: 1,
            nodes: 0,
            time_ms: 0,
            bound: BoundKind::Exact,
            pv: vec![mv],
        });
    }

    let manager = TimeManager::new(Limits::Depth(depth as i32), child.fullmove_number(), 0);
    threads.execute_searches(manager.clone(), Report::None, 1, &child, shared);
    let nodes = shared.nodes.aggregate();
    let time_ms = manager.elapsed().as_millis();
    let result = &threads[0].root_moves[0];
    if result.score == -Score::INFINITE && result.display_score == -Score::INFINITE {
        return None;
    }
    let display = if result.display_score == -Score::INFINITE { result.score } else { result.display_score };
    let bound = if result.lowerbound {
        BoundKind::Upper
    } else if result.upperbound {
        BoundKind::Lower
    } else {
        BoundKind::Exact
    };
    let mut pv = Vec::with_capacity(result.pv.line().len() + 2);
    pv.push(mv);
    pv.push(result.mv);
    pv.extend_from_slice(result.pv.line());
    let (score, mate) = root_score(display, &child);
    Some(BranchSearch {
        score,
        mate,
        seldepth: result.sel_depth.max(0) as usize + 1,
        nodes,
        time_ms,
        bound,
        pv,
    })
}

fn print_info(branches: &[Branch], board: &Board, depth: usize, total_nodes: u64, elapsed_ms: u128) {
    let selected = &branches[selected_index(branches)];
    let seldepth = branches.iter().map(|branch| branch.seldepth).max().unwrap_or(1).max(depth);
    let nps = total_nodes as u128 * 1_000 / elapsed_ms.max(1);
    let score = match selected.mate {
        Some(mate) => format!("mate {mate}"),
        None => format!("cp {}", selected.score),
    };
    let bound = match selected.bound {
        BoundKind::Exact => "",
        BoundKind::Lower => " lowerbound",
        BoundKind::Upper => " upperbound",
    };
    print!(
        "info depth {} seldepth {} multipv 1 score {}{} nodes {} time {} nps {} pv",
        depth, seldepth, score, bound, total_nodes, elapsed_ms, nps
    );
    for mv in &selected.pv {
        print!(" {}", mv.to_uci(board));
    }
    println!();
}

fn best_legal_computation(logits: &[f32], mask: &[bool], candidates: usize) -> Option<usize> {
    logits[..candidates]
        .iter()
        .enumerate()
        .filter(|(index, _)| mask[*index])
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

pub fn go(
    runtime: &Runtime, threads: &mut ThreadPool, board: &Board, shared: &Arc<SharedContext>, limits: Limits,
    move_overhead: u64,
) {
    let mut legal =
        board.generate_all_moves().iter().map(|entry| (entry.mv.to_uci(board), entry.mv)).collect::<Vec<_>>();
    legal.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    if legal.is_empty() {
        println!("info depth 0 score {} 0", if board.in_check() { "mate" } else { "cp" });
        println!("bestmove (none)");
        return;
    }

    let requested_depth = match &limits {
        Limits::Depth(depth) => Some((*depth).max(1) as usize),
        _ => None,
    };
    let requested_mate = match &limits {
        Limits::Mate(moves) => Some(*moves as i32),
        _ => None,
    };
    let maximum_root_depth = requested_depth.unwrap_or(runtime.maximum_depth).min(runtime.maximum_depth);
    let child_depth_limit = maximum_root_depth.saturating_sub(1);
    let mut branches = legal
        .into_iter()
        .map(|(name, mv)| Branch {
            mv,
            name,
            score: static_score(threads, board, mv),
            mate: None,
            depth: 0,
            seldepth: 1,
            cumulative_nodes: 0,
            last_nodes: 0,
            cumulative_time_ms: 0,
            last_score_delta: 0,
            bound: BoundKind::Exact,
            searches: 0,
            pv: vec![mv],
        })
        .collect::<Vec<_>>();
    let initial_budget = runtime.budget;
    let mut remaining_budget = initial_budget;
    let mut total_nodes = 0_u64;
    let mut total_time_ms = 0_u128;
    let mut decision_changes = 0;
    let outer_time = TimeManager::new(limits, board.fullmove_number(), move_overhead);
    shared.status.set(Status::RUNNING);
    println!("info string CS search enabled horizon {initial_budget} maxdepth {maximum_root_depth} mode rolling");
    let mut reported_depth = 1;
    print_info(&branches, board, reported_depth, total_nodes, outer_time.elapsed().as_millis());
    let initial_selected = selected_index(&branches);
    let mut last_report = (initial_selected, branches[initial_selected].depth, branches[initial_selected].score);
    let mut last_report_ms = 0_u128;

    loop {
        let selected = &branches[selected_index(&branches)];
        let depth_reached = requested_depth.is_some_and(|_| selected.depth + 1 >= maximum_root_depth);
        let mate_reached =
            requested_mate.is_some_and(|moves| selected.mate.is_some_and(|mate| mate != 0 && mate.abs() <= moves));
        if depth_reached
            || mate_reached
            || outer_time.hard_limit_reached(total_nodes)
            || shared.externally_stopped.load(Ordering::Acquire)
        {
            break;
        }
        let refresh_incumbent = remaining_budget == 0;
        if refresh_incumbent {
            remaining_budget = initial_budget;
        }
        let previous = selected_index(&branches);
        let (candidates, globals, mask) = encode(
            &branches,
            remaining_budget,
            initial_budget,
            child_depth_limit,
            total_nodes,
            total_time_ms,
            decision_changes,
        );
        let logits = runtime.actor.masked_logits(&candidates, &globals, &mask, branches.len());
        let proposed = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap();
        // UCI limits own termination. If the episodic actor proposes STOP,
        // condition it on continuing and take its best legal computation.
        let incumbent = selected_index(&branches);
        let action = if refresh_incumbent && mask[incumbent] {
            Some(incumbent)
        } else if proposed == branches.len() {
            best_legal_computation(&logits, &mask, branches.len())
        } else {
            Some(proposed)
        };
        let Some(action) = action else {
            break;
        };

        let depth = branches[action].searches + 1;
        let Some(searched) = search_branch(threads, board, shared, branches[action].mv, depth) else {
            break;
        };
        let branch = &mut branches[action];
        let old_score = branch.score;
        branch.score = searched.score;
        branch.mate = searched.mate;
        branch.depth = depth;
        // Native selective depth can fluctuate between independent branch
        // searches. UCI analysis should only advertise the high-water mark.
        branch.seldepth = branch.seldepth.max(searched.seldepth);
        branch.searches += 1;
        branch.last_score_delta = branch.score - old_score;
        branch.last_nodes = searched.nodes;
        branch.cumulative_nodes += searched.nodes;
        branch.cumulative_time_ms += searched.time_ms;
        branch.bound = searched.bound;
        branch.pv = searched.pv;
        total_nodes += searched.nodes;
        total_time_ms += searched.time_ms;
        remaining_budget -= 1;
        if selected_index(&branches) != previous {
            decision_changes += 1;
        }
        if !shared.externally_stopped.load(Ordering::Acquire) {
            shared.status.set(Status::RUNNING);
        }

        let now_ms = outer_time.elapsed().as_millis();
        let current = selected_index(&branches);
        reported_depth = reported_depth.max(branches[current].depth + 1);
        let report = (current, branches[current].depth, branches[current].score);
        if report != last_report || now_ms.saturating_sub(last_report_ms) >= INFO_INTERVAL_MS {
            print_info(&branches, board, reported_depth, total_nodes, now_ms);
            last_report = report;
            last_report_ms = now_ms;
        }
    }

    print_info(&branches, board, reported_depth, total_nodes, outer_time.elapsed().as_millis());
    let selected = &branches[selected_index(&branches)];
    println!("bestmove {}", selected.name);
    shared.status.set(Status::STOPPED);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_actor_loads_and_scores_a_frontier() {
        let runtime = Runtime::default();
        let candidates = vec![0.0; 3 * CANDIDATE_FEATURES];
        let globals = [0.0; GLOBAL_FEATURES];
        let logits = runtime.actor.masked_logits(&candidates, &globals, &[true, false, true, true], 3);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[1], f32::MIN);
    }
}
