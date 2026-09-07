use std::{
    env,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use reckless_cs_burn::{CANDIDATE_FEATURES, GLOBAL_FEATURES, Inference};

const MAX_CANDIDATES: usize = 256;

fn read_f32s(reader: &mut impl Read, count: usize, bytes: &mut Vec<u8>) -> io::Result<Vec<f32>> {
    bytes.resize(count * 4, 0);
    reader.read_exact(bytes)?;
    Ok(bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect())
}

fn serve(weights: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let inference = Inference::load(weights)?;
    let mut reader = BufReader::with_capacity(1 << 20, io::stdin().lock());
    let mut writer = BufWriter::with_capacity(1 << 20, io::stdout().lock());
    let mut header = [0_u8; 4];
    let mut bytes = Vec::new();
    loop {
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        let count = u32::from_le_bytes(header) as usize;
        if count == 0 || count > MAX_CANDIDATES {
            return Err(format!("invalid candidate count {count}").into());
        }
        let candidates = read_f32s(&mut reader, count * CANDIDATE_FEATURES, &mut bytes)?;
        let globals = read_f32s(&mut reader, GLOBAL_FEATURES, &mut bytes)?;
        let mut mask = vec![0_u8; count + 1];
        reader.read_exact(&mut mask)?;
        let mask: Vec<bool> = mask.into_iter().map(|value| value != 0).collect();
        for logit in inference.masked_logits(&candidates, &globals, &mask, count) {
            writer.write_all(&logit.to_le_bytes())?;
        }
        writer.flush()?;
    }
    Ok(())
}

fn benchmark(weights: &Path, count: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    if count == 0 || count > MAX_CANDIDATES {
        return Err(format!("invalid candidate count {count}").into());
    }
    let inference = Inference::load(weights)?;
    let candidates = vec![0.125; count * CANDIDATE_FEATURES];
    let globals = vec![0.25; GLOBAL_FEATURES];
    for _ in 0..100 {
        std::hint::black_box(inference.logits(&candidates, &globals, count));
    }
    let started = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(inference.logits(&candidates, &globals, count));
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({
            "runtime": "rust-burn-flex-f32",
            "burn_version": "0.21.0",
            "resident_weights": true,
            "critic_excluded": true,
            "candidates_per_decision": count,
            "iterations": iterations,
            "seconds": seconds,
            "decisions_per_second": iterations as f64 / seconds,
            "microseconds_per_decision": seconds * 1_000_000.0 / iterations as f64,
            "candidate_logits_per_second": iterations as f64 * count as f64 / seconds,
        })
    );
    Ok(())
}

fn parse_usize(value: Option<String>, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("missing {name}"))?.parse::<usize>().map_err(Into::into)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or("usage: reckless-cs-burn <serve|benchmark> <weights> [arguments]")?;
    let weights = PathBuf::from(args.next().ok_or("missing weights path")?);
    match command.as_str() {
        "serve" => serve(&weights),
        "benchmark" => {
            benchmark(&weights, parse_usize(args.next(), "candidate count")?, parse_usize(args.next(), "iterations")?)
        }
        _ => Err(format!("unknown command {command}").into()),
    }
}
