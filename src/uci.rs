#[cfg(not(any(feature = "cs-search", feature = "physarum-search")))]
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::{
    board::{Board, NullBoardObserver},
    search::Report,
    thread::{SharedContext, Status, ThreadData},
    threadpool::ThreadPool,
    time::Limits,
    tools,
    transposition::DEFAULT_TT_SIZE,
    types::{Color, MAX_MOVES, Piece, Square},
};

#[cfg(not(any(feature = "cs-search", feature = "physarum-search")))]
use crate::{
    time::TimeManager,
    types::{Move, Score, is_decisive, is_loss, is_win},
};

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Cli,
    Uci,
}

struct Settings {
    frc: bool,
    multi_pv: usize,
    move_overhead: u64,
    report: Report,
    #[cfg(feature = "cs-search")]
    cs: crate::cs_search::Runtime,
    #[cfg(feature = "physarum-search")]
    physarum: crate::physarum_search::Runtime,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frc: false,
            multi_pv: 1,
            move_overhead: 100,
            report: Report::Full,
            #[cfg(feature = "cs-search")]
            cs: crate::cs_search::Runtime::default(),
            #[cfg(feature = "physarum-search")]
            physarum: crate::physarum_search::Runtime::default(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn message_loop(mut buffer: VecDeque<String>) {
    let shared = Arc::new(SharedContext::default());
    let mut settings = Settings::default();
    let mut threads = ThreadPool::new(shared.clone());
    let mut board = Board::starting_position();

    let rx = spawn_listener(shared.clone());

    let mut mode = if buffer.is_empty() { Mode::Uci } else { Mode::Cli };

    loop {
        let message = if let Some(cmd) = buffer.pop_front() {
            cmd
        } else if mode == Mode::Uci {
            match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        } else {
            break;
        };

        let tokens = message.split_whitespace().collect::<Vec<_>>();
        match tokens.as_slice() {
            ["uci"] => {
                uci();
                mode = Mode::Uci;
            }

            ["isready"] => println!("readyok"),

            ["go", tokens @ ..] => go(&mut threads, &settings, &board, &shared, tokens),
            ["position", tokens @ ..] => position(&mut board, &settings, tokens),
            ["setoption", tokens @ ..] => set_option(&mut threads, &mut settings, &shared, tokens),
            ["ucinewgame"] => reset(&mut threads, &shared),

            ["stop"] => shared.status.set(Status::STOPPED),
            ["quit"] => {
                drop(threads);
                break;
            }

            // Non-UCI commands
            ["compiler"] => compiler(),
            ["eval"] => eval(threads.main_thread(), &board),
            ["staticeval"] => static_eval(threads.main_thread(), &board),
            ["qeval", nodes] => match nodes.parse::<u64>() {
                Ok(limit) if limit > 0 => quiescence_eval(threads.main_thread(), &board, limit),
                _ => println!("info string qeval requires a positive node cap"),
            },
            ["legalmoves"] => legal_moves(&board),
            ["fen"] => println!("fen {}", board.to_fen()),
            ["d"] => println!("{board}"),
            ["bench", args @ ..] => match mode {
                Mode::Uci => tools::bench::<true>(args),
                Mode::Cli => tools::bench::<false>(args),
            },
            ["speedtest", args @ ..] => tools::speedtest(args),
            ["perft", depth] => tools::perft(depth.parse().unwrap(), &mut board),
            ["perft"] => eprintln!("Usage: perft <depth>"),
            ["simpleperft", depth] => tools::simple_perft(depth.parse().unwrap(), &mut board),
            ["simpleperft"] => eprintln!("Usage: simpleperft <depth>"),
            ["islegalperft", depth] => tools::is_legal_perft(depth.parse().unwrap(), &mut board),
            ["islegalperft"] => eprintln!("Usage: islegalperft <depth>"),

            // Ignore empty lines
            [] => (),

            _ => eprintln!("Unknown command: '{}'", message.trim_end()),
        }

        // Auto-exit after last CLI command
        if matches!(mode, Mode::Cli) && buffer.is_empty() {
            drop(threads);
            break;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_listener(shared: Arc<SharedContext>) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        loop {
            let mut message = String::new();

            if std::io::stdin().read_line(&mut message).unwrap() == 0 {
                // EOF received
                shared.externally_stopped.store(true, Ordering::Release);
                shared.status.set(Status::STOPPED);
                let _ = tx.send("quit".to_string());
                break;
            }

            match message.trim_end() {
                "isready" => println!("readyok"),
                "stop" => {
                    shared.externally_stopped.store(true, Ordering::Release);
                    shared.status.set(Status::STOPPED);
                }
                "quit" => {
                    shared.externally_stopped.store(true, Ordering::Release);
                    shared.status.set(Status::STOPPED);
                    let _ = tx.send("quit".to_string());
                    break;
                }
                _ => {
                    // According to the UCI specs, commands that are unexpected
                    // in the current state should be ignored silently.
                    // (https://backscattering.de/chess/uci/#unexpected)
                    if shared.status.get() != Status::RUNNING {
                        let _ = tx.send(message);
                    }
                }
            }
        }
    });

    rx
}

fn uci() {
    println!("id name Reckless {}", env!("ENGINE_VERSION"));
    println!("id author Arseniy Surkov, Shahin M. Shahin, and Styx");
    println!("option name Hash type spin default {DEFAULT_TT_SIZE} min 1 max 262144");
    println!("option name Threads type spin default 1 min 1 max {}", ThreadPool::available_threads());
    println!("option name MoveOverhead type spin default 100 min 0 max 2000");
    println!("option name Minimal type check default false");
    println!("option name Clear Hash type button");
    println!("option name UCI_Chess960 type check default false");
    println!("option name MultiPV type spin default 1 min 1 max {MAX_MOVES}");

    #[cfg(feature = "cs-search")]
    {
        println!("option name CSBudget type spin default 32 min 1 max 64");
        println!("option name CSMaxDepth type spin default 64 min 1 max 240");
    }

    #[cfg(feature = "physarum-search")]
    {
        println!("option name PhysarumBatch type spin default 64 min 1 max 64");
        println!("option name PhysarumMaxDepth type spin default 64 min 1 max 240");
        println!("option name PhysarumDiagnosticPriorMove type string default none");
        println!("option name PhysarumDiagnosticPriorMass type spin default 990 min 1 max 999");
    }

    #[cfg(feature = "syzygy")]
    println!("option name SyzygyPath type string default");

    #[cfg(feature = "spsa")]
    crate::parameters::print_options();

    println!("uciok");
}

fn compiler() {
    println!("Compiler Version: {}", env!("COMPILER_VERSION"));
    println!("Compiler Target: {}", env!("COMPILER_TARGET"));
    println!("Compiler Features: {}", env!("COMPILER_FEATURES"));
}

fn reset(threads: &mut ThreadPool, shared: &Arc<SharedContext>) {
    threads.clear();
    shared.tt.clear(threads.len());

    for corrhist in shared.history.all() {
        corrhist.pawn.clear();
        corrhist.non_pawn[Color::White].clear();
        corrhist.non_pawn[Color::Black].clear();
    }
}

fn go(threads: &mut ThreadPool, settings: &Settings, board: &Board, shared: &Arc<SharedContext>, tokens: &[&str]) {
    shared.externally_stopped.store(false, Ordering::Release);
    let limits = parse_limits(board.side_to_move(), tokens);

    #[cfg(feature = "cs-search")]
    {
        crate::cs_search::go(&settings.cs, threads, board, shared, limits, settings.move_overhead);
    }

    #[cfg(feature = "physarum-search")]
    {
        crate::physarum_search::go(&settings.physarum, threads, board, shared, limits, settings.move_overhead);
    }

    #[cfg(not(any(feature = "cs-search", feature = "physarum-search")))]
    native_go(threads, settings, board, shared, limits);
}

#[cfg(not(any(feature = "cs-search", feature = "physarum-search")))]
fn native_go(
    threads: &mut ThreadPool, settings: &Settings, board: &Board, shared: &Arc<SharedContext>, limits: Limits,
) {
    let time_manager = TimeManager::new(limits, board.fullmove_number(), settings.move_overhead);

    threads.execute_searches(time_manager, settings.report, settings.multi_pv, board, shared);

    if threads[0].root_moves.is_empty() {
        println!("bestmove (none)");
        return;
    }

    let min_score = threads.iter().map(|v| v.root_moves[0].score).min().unwrap();
    let vote_value = |td: &ThreadData| (td.root_moves[0].score - min_score + 10) * td.completed_depth;

    let mut votes: HashMap<&Move, i32> = HashMap::new();
    for result in threads.iter() {
        *votes.entry(&result.root_moves[0].mv).or_default() += vote_value(result);
    }

    let mut best = 0;

    if !matches!(threads[best].time_manager.limits(), Limits::Depth(_)) && threads[0].multi_pv == 1 {
        for current in 1..threads.len() {
            let is_better_candidate = || -> bool {
                let best = &threads[best];
                let current = &threads[current];

                if is_win(best.root_moves[0].score) {
                    return current.root_moves[0].score > best.root_moves[0].score;
                }

                if current.root_moves[0].score != -Score::INFINITE
                    && best.root_moves[0].score != -Score::INFINITE
                    && is_loss(best.root_moves[0].score)
                {
                    return current.root_moves[0].score < best.root_moves[0].score;
                }

                if current.root_moves[0].score != -Score::INFINITE && is_decisive(current.root_moves[0].score) {
                    return true;
                }

                let best_vote = votes[&best.root_moves[0].mv];
                let current_vote = votes[&current.root_moves[0].mv];

                !is_loss(current.root_moves[0].score)
                    && (current_vote > best_vote
                        || (current_vote == best_vote && vote_value(current) > vote_value(best)))
            };

            if is_better_candidate() {
                best = current;
            }
        }
    }

    if best != 0 {
        let depth = threads[best].completed_depth;
        threads[best].print_uci_info(depth);
    }

    println!("bestmove {}", threads[best].root_moves[0].mv.to_uci(board));
    crate::misc::dbg_print();
}

fn position(board: &mut Board, settings: &Settings, mut tokens: &[&str]) {
    while !tokens.is_empty() {
        match tokens {
            ["startpos", rest @ ..] => {
                *board = Board::starting_position();
                tokens = rest;
            }
            ["fen", rest @ ..] => {
                match Board::from_fen(&rest.join(" ")) {
                    Ok(b) => *board = b,
                    Err(e) => eprintln!("Invalid FEN: {e:?}"),
                }
                board.set_frc(settings.frc);
                tokens = rest;
            }
            ["moves", rest @ ..] => {
                for uci_move in rest {
                    make_uci_move(board, uci_move);
                }
                break;
            }
            _ => tokens = &tokens[1..],
        }
    }
}

fn make_uci_move(board: &mut Board, uci_move: &str) {
    let moves = board.generate_all_moves();
    if let Some(mv) = moves.iter().map(|entry| entry.mv).find(|mv| mv.to_uci(board) == uci_move) {
        board.make_move(mv, &mut NullBoardObserver);
    }
}

fn set_option(threads: &mut ThreadPool, settings: &mut Settings, shared: &Arc<SharedContext>, tokens: &[&str]) {
    match tokens {
        ["name", "Minimal", "value", v] => match *v {
            "true" => settings.report = Report::Minimal,
            "false" => settings.report = Report::Full,
            _ => eprintln!("Invalid value: '{v}'"),
        },
        ["name", "Clear", "Hash"] => {
            shared.tt.clear(threads.len());
            println!("info string Hash cleared");
        }
        ["name", "Hash", "value", v] => {
            shared.tt.resize(threads.len(), v.parse().unwrap());
            println!("info string set Hash to {v} MB");
        }
        ["name", "Threads", "value", v] => {
            threads.set_count(v.parse().unwrap_or(1));
            println!("info string set Threads to {}", threads.len());
        }
        ["name", "MoveOverhead", "value", v] => {
            settings.move_overhead = v.parse().unwrap();
            println!("info string set MoveOverhead to {v} ms");
        }
        #[cfg(feature = "syzygy")]
        ["name", "SyzygyPath", "value", v] => match crate::tb::initialize(v) {
            Some(size) => println!("info string Loaded Syzygy tablebases with {size} pieces"),
            None => eprintln!("Failed to load Syzygy tablebases"),
        },
        ["name", "UCI_Chess960", "value", v] => {
            settings.frc = v.parse().unwrap_or_default();
            println!("info string set UCI_Chess960 to {v}");
        }
        ["name", "MultiPV", "value", v] => {
            settings.multi_pv = v.parse().unwrap_or_default();
            println!("info string set MultiPV to {v}");
        }
        #[cfg(feature = "cs-search")]
        ["name", "CSBudget", "value", v] => {
            settings.cs.set_budget(v);
            println!("info string set CSBudget to {}", settings.cs.budget);
        }
        #[cfg(feature = "cs-search")]
        ["name", "CSMaxDepth", "value", v] => {
            settings.cs.set_maximum_depth(v);
            println!("info string set CSMaxDepth to {}", settings.cs.maximum_depth);
        }
        #[cfg(feature = "physarum-search")]
        ["name", "PhysarumBatch", "value", v] => {
            settings.physarum.set_batch_size(v);
            println!("info string set PhysarumBatch to {}", settings.physarum.batch_size);
        }
        #[cfg(feature = "physarum-search")]
        ["name", "PhysarumMaxDepth", "value", v] => {
            settings.physarum.set_maximum_depth(v);
            println!("info string set PhysarumMaxDepth to {}", settings.physarum.maximum_depth);
        }
        #[cfg(feature = "physarum-search")]
        ["name", "PhysarumDiagnosticPriorMove", "value", v] => {
            settings.physarum.set_diagnostic_prior_move(v);
            println!("info string set PhysarumDiagnosticPriorMove to {v}");
        }
        #[cfg(feature = "physarum-search")]
        ["name", "PhysarumDiagnosticPriorMass", "value", v] => {
            settings.physarum.set_diagnostic_prior_mass(v);
            println!(
                "info string set PhysarumDiagnosticPriorMass to {}",
                settings.physarum.diagnostic_prior_mass_permille
            );
        }
        #[cfg(feature = "spsa")]
        ["name", name, "value", v] => {
            crate::parameters::set_parameter(name, v);
            println!("info string set {name} to {v}");
        }
        _ => eprintln!("Unknown option: '{}'", tokens.join(" ").trim_end()),
    }
}

fn eval(td: &mut ThreadData, board: &Board) {
    td.nnue.full_refresh(board);
    td.nnue.evaluate(board);

    let side = board.side_to_move();

    println!("NNUE derived piece values:");
    println!("+-------+-------+-------+-------+-------+-------+-------+-------+");
    for rank in (0..8).rev() {
        print!("|");
        for file in 0..8 {
            let sq = Square::from_rank_file(rank, file);
            let piece = board.piece_on(sq);
            let piece_str = if piece == Piece::None { " ".to_string() } else { piece.to_string() };
            print!("  {piece_str:^3}  |");
        }
        println!();

        print!("|");
        for file in 0..8 {
            let sq = Square::from_rank_file(rank, file);
            match td.nnue.piece_contribution(board, sq) {
                None => print!("       |"),
                Some(v) => {
                    let white_relative = if side == Color::White { v } else { -v };
                    let val = white_relative as f32 / 100.0;
                    print!("{val:+6.2} |");
                }
            }
        }
        println!();
        println!("+-------+-------+-------+-------+-------+-------+-------+-------+");
    }

    let used_bucket = crate::nnue::OUTPUT_BUCKETS_LAYOUT[board.occupancies().popcount()];

    println!("\nNNUE output buckets (White's POV):");
    println!("+-------------+------------+");
    println!("|   Buckets   |   Total    |");
    println!("+-------------+------------+");

    for bucket in 0..8 {
        let raw_score = td.nnue.eval_with_bucket(board, bucket);
        let white_score = if side == Color::White { raw_score } else { -raw_score };
        let total = white_score as f32 / 100.0;

        if bucket == used_bucket {
            println!("|  >   {bucket:<7}| {total:+7.2}    |");
        } else {
            println!("|{bucket:^13}| {total:+7.2}    |");
        }
    }
    println!("+-------------+------------+");

    let final_eval = td.nnue.evaluate(board);
    let final_total = (if side == Color::White { final_eval } else { -final_eval }) as f32 / 100.0;
    println!("\nNNUE evaluation        {final_total:+.2} (White's POV)");
}

/// Print only the frozen NNUE score, from the side-to-move perspective.
///
/// This command is intentionally search-free.  It exists for controlled
/// experiments that use Reckless solely as a leaf evaluator without exposing
/// or imitating the engine's native search and move-ordering machinery.
fn static_eval(td: &mut ThreadData, board: &Board) {
    td.nnue.full_refresh(board);
    println!("staticeval {}", td.nnue.evaluate(board));
}

/// Experimental full-window quiescence, with the same score units as UCI search.
fn quiescence_eval(td: &mut ThreadData, board: &Board, limit: u64) {
    use crate::types::{Score, normalize_to_cp};
    td.time_manager = crate::time::TimeManager::new(Limits::Infinite, 0, 0);
    td.shared.nodes.reset();
    td.shared.status.set(Status::RUNNING);
    td.qeval_node_limit = Some(limit);
    td.qeval_truncated = false;
    let score = crate::search::quiescence_evaluate(td, board);
    let truncated = u8::from(td.qeval_truncated);
    td.qeval_node_limit = None;
    td.shared.status.set(Status::STOPPED);
    let nodes = 1 + td.shared.nodes.aggregate();
    if score.abs() >= Score::MATE_IN_MAX {
        let mate = (Score::MATE - score.abs() + score.is_positive() as i32) / 2;
        println!("qeval mate {} nodes {nodes} truncated {truncated}", if score.is_positive() { mate } else { -mate });
    } else if score.abs() >= Score::TB_WIN_IN_MAX {
        let cp = 20_000 - Score::TB_WIN + score.abs();
        println!("qeval cp {} nodes {nodes} truncated {truncated}", if score.is_positive() { cp } else { -cp });
    } else {
        println!("qeval cp {} nodes {nodes} truncated {truncated}", normalize_to_cp(score, board));
    }
}

/// Enumerate legal moves with Reckless's native move generator.
///
/// Keeping this command search-free lets computation-allocation experiments
/// discover their legal root actions without maintaining a second chess move
/// generator outside the engine.
fn legal_moves(board: &Board) {
    println!("legalmoves {}", legal_move_strings(board).join(" "));
}

fn legal_move_strings(board: &Board) -> Vec<String> {
    let mut moves = board.generate_all_moves().iter().map(|entry| entry.mv.to_uci(board)).collect::<Vec<_>>();
    moves.sort_unstable();
    moves
}

fn parse_limits(color: Color, tokens: &[&str]) -> Limits {
    if let ["infinite"] = tokens {
        return Limits::Infinite;
    }

    let mut main = None;
    let mut inc = None;
    let mut moves = None;

    for chunk in tokens.chunks(2) {
        if let [name, value] = *chunk {
            let Ok(value) = value.parse::<u64>() else {
                continue;
            };

            match name {
                "depth" if value > 0 => return Limits::Depth(value as i32),
                "movetime" if value > 0 => return Limits::Time(value),
                "nodes" if value > 0 => return Limits::Nodes(value),
                "mate" if value > 0 => return Limits::Mate(value),

                "wtime" if Color::White == color => main = Some(value),
                "btime" if Color::Black == color => main = Some(value),
                "winc" if Color::White == color => inc = Some(value),
                "binc" if Color::Black == color => inc = Some(value),
                "movestogo" => moves = Some(value),

                _ => continue,
            }
        }
    }

    if main.is_none() && inc.is_none() {
        return Limits::Infinite;
    }

    let main = main.unwrap_or_default();
    let inc = inc.unwrap_or_default();

    match moves {
        Some(moves) => Limits::Cyclic(main, inc, moves),
        None => Limits::Fischer(main, inc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_position_helper(tokens: &[&str]) -> Board {
        let settings = Settings::default();
        let mut board = Board::starting_position();

        position(&mut board, &settings, tokens);
        board.clone()
    }

    #[test]
    fn test_position_startpos() {
        let board = test_position_helper(&["startpos"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let board = test_position_helper(&[]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn test_position_startpos_multiple_moves() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5", "g1f3"]);
        assert_eq!(board.side_to_move(), Color::Black);
        let fen = board.to_fen();
        let fen_position = fen.split_whitespace().next().unwrap();
        assert!(fen_position.contains("5N2"));
    }

    #[test]
    fn test_position_fen_with_moves() {
        let board = test_position_helper(&[
            "fen",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
            "b",
            "KQkq",
            "e3",
            "0",
            "1",
            "moves",
            "e7e5",
        ]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_position_empty_moves_list() {
        let board = test_position_helper(&["moves"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn test_position_invalid_move_ignored() {
        let board = test_position_helper(&["moves", "e2e4", "invalid", "e7e5"]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_legal_move_strings_uses_native_generator() {
        let board = Board::starting_position();
        let moves = legal_move_strings(&board);
        assert_eq!(moves.len(), 20);
        assert!(moves.iter().any(|mv| mv == "e2e4"));
        assert!(moves.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn test_native_fen_after_move() {
        let mut board = Board::starting_position();
        make_uci_move(&mut board, "e2e4");
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    }

    #[test]
    fn test_position_long_move_sequence() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_position_castling() {
        let board = test_position_helper(&[
            "fen",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R",
            "w",
            "KQkq",
            "-",
            "0",
            "1",
            "moves",
            "e1g1",
        ]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_position_en_passant() {
        let board = test_position_helper(&[
            "fen",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR",
            "w",
            "KQkq",
            "f6",
            "0",
            "1",
            "moves",
            "e5f6",
        ]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_position_promotion() {
        let board = test_position_helper(&["fen", "8/P7/8/8/8/8/8/4K2k", "w", "-", "-", "0", "1", "moves", "a7a8q"]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_make_uci_move_invalid() {
        let mut board = Board::starting_position();
        let fen_before = board.to_fen();
        make_uci_move(&mut board, "invalid_move");
        assert_eq!(board.to_fen(), fen_before);
    }

    #[test]
    fn test_position_moves_without_startpos_ignored() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    }
}
