//! Amortized CPU inference for the trained, permutation-equivariant edge head.
//! The 914 board/move features are invariant during a tree search and their
//! first-layer contribution is cached per edge. Only ten observed statistics
//! and the remaining small layers are recomputed when the flow changes.

use crate::{
    board::Board,
    types::{CastlingKind, Color, Move, Piece, Square},
};

const INPUT: usize = 924;
const STATIC: usize = 914;
const MAGIC: &[u8; 8] = b"PCNDv2\0\0";

#[derive(Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn from_reader(reader: &mut Reader<'_>, input: usize, output: usize) -> Result<Self, String> {
        Ok(Self {
            weight: reader.floats(input * output)?,
            bias: reader.floats(output)?,
            input,
            output,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.input);
        let mut result = self.bias.clone();
        for (row, value) in self.weight.chunks_exact(self.input).zip(&mut result) {
            for (weight, input) in row.iter().zip(x) {
                *value += weight * input;
            }
        }
        result
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let bytes: [u8; 4] =
            self.bytes.get(self.offset..self.offset + 4).ok_or("truncated conductivity header")?.try_into().unwrap();
        self.offset += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let end = self
            .offset
            .checked_add(count.checked_mul(4).ok_or("invalid conductivity shape")?)
            .ok_or("invalid conductivity shape")?;
        let bytes = self.bytes.get(self.offset..end).ok_or("truncated conductivity weights")?;
        self.offset = end;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}

#[derive(Clone)]
pub struct Head {
    pub update: u32,
    first: Linear,
    second: Linear,
    hidden: Linear,
    output: Linear,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Head {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.get(..8) != Some(MAGIC.as_slice()) {
            return Err("invalid conductivity model magic".into());
        }
        let mut reader = Reader { bytes, offset: 8 };
        let width = reader.u32()? as usize;
        let update = reader.u32()?;
        if !(1..=512).contains(&width) {
            return Err("invalid conductivity width".into());
        }
        let first = Linear::from_reader(&mut reader, INPUT, width)?;
        let second = Linear::from_reader(&mut reader, width, width)?;
        let hidden = Linear::from_reader(&mut reader, 2 * width, width)?;
        let output = Linear::from_reader(&mut reader, width, 1)?;
        if reader.offset != bytes.len() {
            return Err("trailing conductivity model bytes".into());
        }
        Ok(Self { update, first, second, hidden, output })
    }

    pub fn static_embedding(&self, board: &Board, mv: Move) -> Vec<f32> {
        // The first layer is linear. Adding only nonzero sparse static columns
        // avoids 924 multiplies per neuron per edge on every flow step.
        let mut columns = Vec::with_capacity(50);
        for square in 0..64 {
            let sq = Square::new(square);
            let piece = board.piece_on(sq);
            if piece != Piece::None {
                let plane = (if piece.color() == Color::White { 0 } else { 6 }) + piece.piece_type() as usize;
                columns.push(plane * 64 + square as usize);
            }
        }
        let side = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
        for (index, kind) in [
            CastlingKind::WhiteKingside,
            CastlingKind::WhiteQueenside,
            CastlingKind::BlackKingside,
            CastlingKind::BlackQueenside,
        ]
        .iter()
        .enumerate()
        {
            if board.castling().is_allowed(*kind) {
                columns.push(769 + index);
            }
        }
        if board.en_passant() != Square::None {
            columns.push(773 + board.en_passant().file() as usize);
        }
        columns.push(781 + mv.from() as usize);
        columns.push(845 + mv.to() as usize);
        let promotion = if mv.is_promotion() { mv.promo_piece_type() as usize } else { 0 };
        columns.push(909 + promotion);
        let mut base = self.first.bias.clone();
        for (neuron, row) in self.first.weight.chunks_exact(INPUT).enumerate() {
            base[neuron] += row[768] * side;
            for &column in &columns {
                base[neuron] += row[column];
            }
        }
        base
    }

    pub fn conductivities(&self, bases: &[&[f32]], statistics: &[[f32; 10]]) -> Vec<f64> {
        debug_assert_eq!(bases.len(), statistics.len());
        let width = self.first.output;
        let mut encoded = Vec::with_capacity(bases.len());
        let mut mean = vec![0.0; width];
        for (base, stats) in bases.iter().zip(statistics) {
            let mut h = base.to_vec();
            for (j, row) in self.first.weight.chunks_exact(INPUT).enumerate() {
                for (weight, stat) in row[STATIC..].iter().zip(stats) {
                    h[j] += weight * stat;
                }
                h[j] = silu(h[j]);
            }
            let h = self.second.forward(&h).into_iter().map(silu).collect::<Vec<_>>();
            for (sum, value) in mean.iter_mut().zip(&h) {
                *sum += *value;
            }
            encoded.push(h);
        }
        for value in &mut mean {
            *value /= bases.len() as f32;
        }
        let mut logits = Vec::with_capacity(bases.len());
        for h in encoded {
            let mut input = h;
            input.extend_from_slice(&mean);
            let hidden = self.hidden.forward(&input).into_iter().map(silu).collect::<Vec<_>>();
            logits.push(self.output.forward(&hidden)[0]);
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials = logits.iter().map(|x| (*x - maximum).exp() as f64).collect::<Vec<_>>();
        let total = exponentials.iter().sum::<f64>();
        exponentials.into_iter().map(|x| 0.01 + x / total).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_loads_and_returns_positive_sibling_conductivity() {
        let head =
            Head::from_bytes(include_bytes!("../experiments/computation_allocation/artifacts/conductivity-u29.bin"))
                .unwrap();
        let board = Board::starting_position();
        let moves: Vec<_> = board.generate_all_moves().iter().map(|entry| entry.mv).collect();
        let bases: Vec<_> = moves.iter().map(|mv| head.static_embedding(&board, *mv)).collect();
        let refs: Vec<_> = bases.iter().map(Vec::as_slice).collect();
        let scores = head.conductivities(&refs, &vec![[0.0; 10]; refs.len()]);
        assert_eq!(scores.len(), moves.len());
        assert!(scores.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!((scores.iter().sum::<f64>() - (1.0 + 0.01 * moves.len() as f64)).abs() < 1e-5);
    }

    #[test]
    fn matches_pytorch_forward_on_three_start_position_moves() {
        let head =
            Head::from_bytes(include_bytes!("../experiments/computation_allocation/artifacts/conductivity-u29.bin"))
                .unwrap();
        let board = Board::starting_position();
        let names = ["e2e4", "d2d4", "g1f3"];
        let legal: Vec<_> = board.generate_all_moves().iter().map(|entry| entry.mv).collect();
        let bases: Vec<_> = names
            .iter()
            .map(|name| {
                let mv = legal.iter().find(|mv| mv.to_uci(&board) == *name).unwrap();
                head.static_embedding(&board, *mv)
            })
            .collect();
        let refs: Vec<_> = bases.iter().map(Vec::as_slice).collect();
        let stats = [[0.25, 0.1, 0.2, 0.2, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0]; 3];
        let actual = head.conductivities(&refs, &stats);
        let expected = [0.34326034784317017, 0.3434358239173889, 0.3433038294315338];
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{got} versus {want}");
        }
    }

    #[test]
    fn matches_pytorch_with_en_passant_and_promotion() {
        let head =
            Head::from_bytes(include_bytes!("../experiments/computation_allocation/artifacts/conductivity-u29.bin"))
                .unwrap();
        let board = Board::from_fen("4k3/P7/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let names = ["e5d6", "a7a8q", "e1e2"];
        let legal: Vec<_> = board.generate_all_moves().iter().map(|entry| entry.mv).collect();
        let bases: Vec<_> = names
            .iter()
            .map(|name| {
                let mv = legal.iter().find(|mv| mv.to_uci(&board) == *name).unwrap();
                head.static_embedding(&board, *mv)
            })
            .collect();
        let refs: Vec<_> = bases.iter().map(Vec::as_slice).collect();
        let stats = [[0.25, 0.1, 0.2, 0.2, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0]; 3];
        let expected = [0.34333592653274536, 0.3433740735054016, 0.3432900309562683];
        for (got, want) in head.conductivities(&refs, &stats).iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{got} versus {want}");
        }
    }
}
