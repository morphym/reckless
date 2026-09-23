//! Native execution of the trained Python Physarum tree algorithm.
//! The head only supplies positive edge conductivities. Evaluated children,
//! minimax backups, and the root value determine the played move.

use std::{
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use crate::{
    board::{Board, NullBoardObserver},
    conductivity_head::Head,
    search::quiescence_evaluate,
    thread::{SharedContext, Status},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    types::{Color, Move, Score, normalize_to_cp},
};

#[derive(Clone)]
struct Node {
    board: Board,
    parent: Option<usize>,
    mv: Option<Move>,
    children: Vec<usize>,
    depth: usize,
    initial: f32,
    value: f32,
    visits: u32,
    deposit: f32,
    frontier: bool,
    terminal: bool,
    complete_expansion: bool,
    prior_conductivity: f64,
}

fn legal_moves(board: &Board) -> Vec<Move> {
    board.generate_all_moves().iter().map(|entry| entry.mv).collect()
}

fn heuristic_conductivities(board: &Board, moves: &[Move]) -> Vec<f64> {
    let scores = moves
        .iter()
        .map(|mv| {
            // Only use stable tactical facts as the non-learned prior.  The
            // native picker also has TT/history/continuation scores, but
            // those are statistical ordering hints, not evidence that a move
            // is good.  In particular, do not bake its list order into the
            // flow: that turns an arbitrary generator order into a false
            // conductivity signal.
            let mut score = 1.0;
            if mv.is_capture() {
                // SEE is a useful gate, not a proof.  Reward a non-losing
                // exchange modestly and suppress obviously losing captures;
                // the quiescence-backed tree must still verify both.
                if board.see(*mv, 0) {
                    score += 1.0;
                } else {
                    score *= 0.35;
                }
            }
            if mv.is_promotion() {
                score += 1.5;
            }
            // `is_direct_check` is deliberately only a tiny nudge: Reckless
            // documents it as an approximate test and checking sacrifices are
            // a classic source of misleading priors.
            if board.is_direct_check(*mv) {
                score += 0.15;
            }
            score
        })
        .collect::<Vec<_>>();
    let total = scores.iter().sum::<f64>();
    scores.into_iter().map(|score| 0.01 + 0.05 * score / total).collect()
}

fn terminal_value(board: &Board, root: Color) -> Option<f32> {
    let moves = legal_moves(board);
    if moves.is_empty() {
        return Some(if board.in_check() { if board.side_to_move() == root { -1.0 } else { 1.0 } } else { 0.0 });
    }
    // Training uses python-chess outcome(claim_draw=False), so the optional
    // fifty-move and threefold claims must not stop the explicit tree.
    if board.draw_by_material() || board.fiftymove_clock() >= 150 { Some(0.0) } else { None }
}

fn evaluate(
    board: &Board, root: Color, qnodes: u64, threads: &mut ThreadPool, shared: &Arc<SharedContext>,
) -> (f32, u64, bool) {
    shared.nodes.reset();
    let td = threads.main_thread();
    td.qeval_node_limit = Some(qnodes);
    td.qeval_truncated = false;
    td.time_manager = TimeManager::new(Limits::Infinite, board.fullmove_number(), 0);
    let score = quiescence_evaluate(td, board);
    let truncated = td.qeval_truncated;
    td.qeval_node_limit = None;
    let nodes = 1 + shared.nodes.aggregate();
    let utility = if score.abs() >= Score::TB_WIN_IN_MAX {
        if score > 0 { 1.0 } else { -1.0 }
    } else {
        (normalize_to_cp(score, board) as f32 / 600.0).tanh()
    };
    (if board.side_to_move() == root { utility } else { -utility }, nodes, truncated)
}

fn backup(nodes: &mut [Node], mut index: usize, root: Color) {
    loop {
        if !nodes[index].children.is_empty() {
            let maximizing = nodes[index].board.side_to_move() == root;
            nodes[index].value = nodes[index].children.iter().map(|child| nodes[*child].value).fold(
                if maximizing { f32::NEG_INFINITY } else { f32::INFINITY },
                |a, b| {
                    if maximizing { a.max(b) } else { a.min(b) }
                },
            );
        }
        match nodes[index].parent {
            Some(parent) => index = parent,
            None => break,
        }
    }
}

fn expand(
    nodes: &mut Vec<Node>, index: usize, root: Color, head: &Head, learned: bool, budget: usize, max_depth: usize,
    qnodes: u64, threads: &mut ThreadPool, shared: &Arc<SharedContext>, evals: &mut usize, qcount: &mut u64,
    truncated: &mut u64, manager: &TimeManager, force: bool,
) -> bool {
    let moves = legal_moves(&nodes[index].board);
    if moves.len() > budget.saturating_sub(*evals) {
        return false;
    }
    let expected_children = moves.len();
    nodes[index].frontier = false;
    let parent_board = nodes[index].board.clone();
    let depth = nodes[index].depth + 1;
    let mut bases = Vec::with_capacity(expected_children);
    for mv in moves {
        // The root expansion is the minimum useful unit of work.  A very
        // short UCI movetime can trip the asynchronous stop flag before the
        // first child is evaluated; honouring it here leaves an empty tree
        // and the caller then plays the first legal move without searching.
        // `force` is only used for this one root expansion, so it cannot make
        // the subsequent tree growth ignore cancellation or time limits.
        if (!force && shared.externally_stopped.load(Ordering::Acquire))
            || (!force && manager.hard_limit_reached(*qcount))
        {
            break;
        }
        let mut board = parent_board.clone();
        board.make_move(mv, &mut NullBoardObserver);
        let terminal = terminal_value(&board, root);
        let value = if let Some(value) = terminal {
            value
        } else {
            let (value, used, cut) = evaluate(&board, root, qnodes, threads, shared);
            *qcount += used;
            *truncated += u64::from(cut);
            value
        };
        let child = nodes.len();
        bases.push(head.static_embedding(&parent_board, mv));
        nodes.push(Node {
            board,
            parent: Some(index),
            mv: Some(mv),
            children: Vec::new(),
            depth,
            initial: value,
            value,
            visits: 0,
            deposit: 0.0,
            frontier: depth < max_depth && terminal.is_none(),
            terminal: terminal.is_some(),
            complete_expansion: false,
            prior_conductivity: 0.0,
        });
        nodes[index].children.push(child);
        *evals += 1;
    }
    nodes[index].complete_expansion = nodes[index].children.len() == expected_children;
    if !bases.is_empty() && learned {
        // The teacher supervises this pre-evidence state. An unexpanded child
        // never receives a fabricated value or a penalty for lacking evidence.
        let inputs = bases.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let stats = vec![[1.0 / max_depth as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; inputs.len()];
        let priors = head.conductivities(&inputs, &stats);
        let assigned = nodes[index].children.iter().copied().zip(priors).collect::<Vec<_>>();
        for (child, prior) in assigned {
            // The policy is an initial hint, not a pre-existing proof.
            nodes[child].prior_conductivity = 0.05 * prior;
        }
    } else if !learned {
        let child_ids = nodes[index].children.clone();
        let priors = heuristic_conductivities(
            &parent_board,
            &child_ids.iter().map(|child| nodes[*child].mv.unwrap()).collect::<Vec<_>>(),
        );
        for (child, prior) in child_ids.into_iter().zip(priors) {
            nodes[child].prior_conductivity = prior;
        }
    }
    backup(nodes, index, root);
    nodes[index].complete_expansion
}

fn flow(nodes: &[Node]) -> Vec<f64> {
    let mut conduct = vec![0.0; nodes.len()];
    for (i, node) in nodes.iter().enumerate().skip(1) {
        conduct[i] = (node.prior_conductivity + f64::from(node.deposit)).max(1e-12);
    }
    let mut effective = vec![0.0; nodes.len()];
    let mut branch = vec![0.0; nodes.len()];
    for i in (0..nodes.len()).rev() {
        if nodes[i].frontier {
            effective[i] = f64::INFINITY;
            continue;
        }
        for &child in &nodes[i].children {
            let downstream = effective[child];
            branch[child] = if downstream.is_infinite() {
                conduct[child]
            } else if downstream > 0.0 {
                conduct[child] * downstream / (conduct[child] + downstream)
            } else {
                0.0
            };
            effective[i] += branch[child];
        }
    }
    let mut currents = vec![0.0; nodes.len()];
    currents[0] = 1.0;
    for (i, node) in nodes.iter().enumerate() {
        if effective[i] <= 0.0 || node.frontier {
            continue;
        }
        for &child in &node.children {
            currents[child] = currents[i] * branch[child] / effective[i];
        }
    }
    currents
}

fn random_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

fn choose_frontier(nodes: &[Node], currents: &[f64], rng: &mut u64) -> Option<usize> {
    let total = nodes.iter().enumerate().filter(|(_, n)| n.frontier).map(|(i, _)| currents[i]).sum::<f64>();
    if total <= 0.0 {
        return None;
    }
    let mut target = random_unit(rng) * total;
    let mut last = None;
    for (i, node) in nodes.iter().enumerate() {
        if !node.frontier || currents[i] <= 0.0 {
            continue;
        }
        last = Some(i);
        target -= currents[i];
        if target <= 0.0 {
            return Some(i);
        }
    }
    last
}

pub fn go(
    head: &Head, learned: bool, budget: usize, max_depth: usize, qnodes: u64, seed: u64, threads: &mut ThreadPool,
    board: &Board, shared: &Arc<SharedContext>, limits: Limits, move_overhead: u64,
) {
    let moves = legal_moves(board);
    if moves.is_empty() {
        println!("bestmove (none)");
        shared.status.set(Status::STOPPED);
        return;
    }
    let max_depth = match limits {
        Limits::Depth(depth) => max_depth.min(depth.max(1) as usize),
        _ => max_depth,
    };
    let manager = TimeManager::new(limits, board.fullmove_number(), move_overhead);
    shared.status.set(Status::RUNNING);
    let started = Instant::now();
    let root = board.side_to_move();
    let mut nodes = vec![Node {
        board: board.clone(),
        parent: None,
        mv: None,
        children: Vec::new(),
        depth: 0,
        initial: 0.0,
        value: 0.0,
        visits: 0,
        deposit: 0.0,
        frontier: true,
        terminal: false,
        complete_expansion: false,
        prior_conductivity: 0.0,
    }];
    let mut evals = 0;
    let mut qcount = 0;
    let mut truncated = 0;
    let mut steps = 0;
    let mut rng = board.hash() ^ seed ^ 0x9e3779b97f4a7c15;
    if rng == 0 {
        rng = 0x9e3779b97f4a7c15;
    }
    if budget < moves.len() {
        println!("info string PhysarumBudget {budget} below root branching {}; using first legal move", moves.len());
        println!("bestmove {}", moves[0].to_uci(board));
        shared.status.set(Status::STOPPED);
        return;
    }
    expand(
        &mut nodes,
        0,
        root,
        head,
        learned,
        budget,
        max_depth,
        qnodes,
        threads,
        shared,
        &mut evals,
        &mut qcount,
        &mut truncated,
        &manager,
        true,
    );
    while evals < budget && !manager.hard_limit_reached(qcount) && !shared.externally_stopped.load(Ordering::Acquire) {
        let currents = flow(&nodes);
        let Some(index) = choose_frontier(&nodes, &currents, &mut rng) else {
            break;
        };
        let old_root = nodes[0].value;
        if !expand(
            &mut nodes,
            index,
            root,
            head,
            learned,
            budget,
            max_depth,
            qnodes,
            threads,
            shared,
            &mut evals,
            &mut qcount,
            &mut truncated,
            &manager,
            false,
        ) {
            nodes[index].frontier = false;
        } else {
            // Deposit is evidence, not surprise.  The old absolute-delta
            // rule reinforced a branch when its backed-up value got worse as
            // well as when it got better; that lets a spectacular blunder
            // (for example a queen sacrifice) attract all later flow.  Values
            // are already expressed from the root player's perspective, so
            // only positive improvements are admissible evidence.  The small
            // floor preserves weak prior exploration without allowing it to
            // overwhelm a supported line.
            let root_gain = (nodes[0].value - old_root).max(0.0);
            let branch_gain = (nodes[index].value - nodes[index].initial).max(0.0);
            // A decisive positive leaf is stronger than an ordinary numeric
            // improvement.  Do not let a high but misleading quiescence
            // baseline dilute a discovered mate/tablebase win: once such
            // evidence is present, route a full-strength deposit along the
            // complete path.  Decisive losses deliberately receive no such
            // bonus and remain available only through the weak prior.
            let decisive_positive = nodes[index].value >= 0.999;
            let utility = if decisive_positive {
                1.0
            } else {
                (0.01 + root_gain + branch_gain).min(1.0)
            };
            for node in &mut nodes {
                node.deposit *= 0.97;
            }
            let depth = nodes[index].depth as f32;
            let mut current = Some(index);
            while let Some(i) = current {
                if i == 0 {
                    break;
                }
                nodes[i].visits += 1;
                nodes[i].deposit += 0.75 * utility / depth;
                current = nodes[i].parent;
            }
        }
        steps += 1;
    }
    let best = *nodes[0]
        .children
        .iter()
        .max_by(|a, b| nodes[**a].value.total_cmp(&nodes[**b].value).then_with(|| b.cmp(a)))
        .unwrap_or(&0);
    let best_move = nodes[best].mv.unwrap_or(moves[0]);
    let cp = ((nodes[best].value.clamp(-0.999, 0.999).atanh() * 600.0) as i32).clamp(-20_000, 20_000);
    let elapsed = started.elapsed().as_millis();
    let seldepth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut completed = vec![0; nodes.len()];
    for i in (0..nodes.len()).rev() {
        completed[i] = if nodes[i].terminal {
            max_depth
        } else if nodes[i].complete_expansion && !nodes[i].children.is_empty() {
            1 + nodes[i].children.iter().map(|child| completed[*child]).min().unwrap()
        } else {
            0
        };
    }
    let depth = completed[0].min(max_depth);
    println!(
        "info depth {depth} seldepth {seldepth} score cp {cp} nodes {qcount} time {elapsed} pv {}",
        best_move.to_uci(board)
    );
    println!(
        "info string learned-physarum update {} frontier-evaluations {evals} qsearch-nodes {qcount} qsearch-truncated {truncated} flow-steps {steps} tree-nodes {}",
        head.update,
        nodes.len()
    );
    println!("bestmove {}", best_move.to_uci(board));
    shared.status.set(Status::STOPPED);
}
