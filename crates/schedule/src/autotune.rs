//! Oracle-gated schedule autotuning.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;
use vtc_interp::{Tensor, TensorF64};
use vtc_loopir::{BufferRole, Kernel, eval_loops, eval_loops_f64};

use crate::primitives::{constant_value, decompose_perfect_nest};
use crate::{CostModel, Mode, fuse, interchange, tile};

/// One schedule move considered by the autotuner.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MoveDesc {
    /// Swap adjacent loop levels in one top-level nest.
    Interchange {
        /// Top-level nest index.
        nest: usize,
        /// First of the two adjacent levels.
        level: usize,
    },
    /// Strip-mine one loop level.
    Tile {
        /// Top-level nest index.
        nest: usize,
        /// Loop level to strip-mine.
        level: usize,
        /// Even tile size.
        size: usize,
    },
    /// Fuse two adjacent top-level nests.
    Fuse {
        /// Producer nest index.
        a: usize,
        /// Consumer nest index.
        b: usize,
    },
}

/// Autotuner configuration.
#[derive(Debug, Clone)]
pub struct TuneConfig {
    /// Legality and validation mode.
    pub mode: Mode,
    /// Candidate tile sizes, tried in listed order.
    pub tile_sizes: Vec<usize>,
    /// Maximum improving rounds per restart attempt.
    pub max_rounds: usize,
    /// Oracle validation trials per candidate.
    pub validation_trials: usize,
    /// Number of bounded restart attempts after the initial greedy run.
    pub restarts: usize,
    /// Seed for deterministic validation input generation.
    pub seed: u64,
}

impl Default for TuneConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Strict,
            tile_sizes: vec![2, 4, 8, 16, 32],
            max_rounds: 16,
            validation_trials: 16,
            restarts: 0,
            seed: 0x7155_0001,
        }
    }
}

/// Result of a tuning run.
#[derive(Debug, Clone)]
pub struct TuneResult {
    /// Best validated kernel found.
    pub kernel: Kernel,
    /// Initial cost.
    pub cost_before: u64,
    /// Final cost.
    pub cost_after: u64,
    /// Adopted moves in order.
    pub moves: Vec<MoveDesc>,
    /// Number of candidates examined by the oracle gate.
    pub candidates_evaluated: usize,
}

/// Errors produced by the autotuner.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TuneError {
    /// Internal consistency check failed.
    #[error("internal autotune error: {0}")]
    Internal(String),
}

/// Enumerates all one-step legal moves for `kernel` under `cfg`.
///
/// Moves are legal by construction because each candidate is produced by the
/// existing legality-checked primitive. Failed primitive calls are ignored.
#[must_use]
pub fn legal_moves(kernel: &Kernel, cfg: &TuneConfig) -> Vec<(MoveDesc, Kernel)> {
    let mut moves = Vec::new();

    for (nest, stmt) in kernel.body().iter().enumerate() {
        if let Ok(perfect) = decompose_perfect_nest(stmt) {
            for level in 0..perfect.levels.len().saturating_sub(1) {
                if let Ok(candidate) = interchange(kernel, nest, level, cfg.mode) {
                    moves.push((MoveDesc::Interchange { nest, level }, candidate));
                }
            }

            for level in 0..perfect.levels.len() {
                for &size in &cfg.tile_sizes {
                    if tile_size_may_divide(&perfect.levels[level], size)
                        && let Ok(candidate) = tile(kernel, nest, level, size, cfg.mode)
                    {
                        moves.push((MoveDesc::Tile { nest, level, size }, candidate));
                    }
                }
            }
        }
    }

    for first in 0..kernel.body().len().saturating_sub(1) {
        let second = first.saturating_add(1);
        if let Ok(candidate) = fuse(kernel, first, second, cfg.mode) {
            moves.push((
                MoveDesc::Fuse {
                    a: first,
                    b: second,
                },
                candidate,
            ));
        }
    }

    moves
}

/// Validates two kernels with differential loop-oracle trials.
///
/// The gate checks exact-rational equivalence for every mode. In
/// [`Mode::Strict`] it also requires `eval_loops_f64` bit identity. This is a
/// testing-strength implementation backstop; the schedule primitives' legality
/// rules remain the soundness argument.
pub fn validate_equiv(
    original: &Kernel,
    candidate: &Kernel,
    mode: Mode,
    trials: usize,
    rng: &mut StdRng,
) -> bool {
    for _ in 0..trials {
        let rational_inputs = random_rational_inputs(original, rng);
        let Ok(left) = eval_loops(original, &rational_inputs) else {
            return false;
        };
        let Ok(right) = eval_loops(candidate, &rational_inputs) else {
            return false;
        };
        if left != right {
            return false;
        }

        if mode == Mode::Strict {
            let f64_inputs = random_f64_inputs(original, rng);
            let Ok(left) = eval_loops_f64(original, &f64_inputs) else {
                return false;
            };
            let Ok(right) = eval_loops_f64(candidate, &f64_inputs) else {
                return false;
            };
            if left
                .iter()
                .zip(&right)
                .any(|(left, right)| !left.bit_eq(right))
            {
                return false;
            }
            if left.len() != right.len() {
                return false;
            }
        }
    }
    true
}

/// Runs a deterministic, bounded greedy schedule search.
///
/// The search itself is untrusted. It only enumerates kernels produced by
/// legality-checked primitives, and it adopts an improving candidate only after
/// the loop-oracle gate validates it against the original input kernel. If no
/// candidate both improves and validates, the current best kernel is returned.
///
/// # Errors
///
/// Returns [`TuneError`] for internal arithmetic or budget inconsistencies.
pub fn autotune(
    kernel: &Kernel,
    cost: &dyn CostModel,
    cfg: &TuneConfig,
) -> Result<TuneResult, TuneError> {
    let start = kernel.clone();
    let cost_before = cost.cost(&start);
    let mut best = start.clone();
    let mut best_cost = cost_before;
    let mut best_moves = Vec::new();
    let mut candidates_evaluated = 0usize;
    let attempts = cfg.restarts.saturating_add(1);

    for restart in 0..attempts {
        let mut current = start.clone();
        let mut current_cost = cost_before;
        let mut moves = Vec::new();
        let mut rng = StdRng::seed_from_u64(
            cfg.seed.wrapping_add(
                u64::try_from(restart)
                    .map_err(|_| TuneError::Internal("restart index overflow".to_owned()))?,
            ),
        );

        for _ in 0..cfg.max_rounds {
            let mut candidates = legal_moves(&current, cfg);
            candidates.sort_by_key(|(desc, candidate)| (cost.cost(candidate), desc.clone()));
            let mut adopted = false;

            for (desc, candidate) in candidates {
                candidates_evaluated = candidates_evaluated.saturating_add(1);
                let candidate_cost = cost.cost(&candidate);
                if candidate_cost >= current_cost {
                    continue;
                }
                if validate_equiv(
                    &start,
                    &candidate,
                    cfg.mode,
                    cfg.validation_trials,
                    &mut rng,
                ) {
                    current = candidate;
                    current_cost = candidate_cost;
                    moves.push(desc);
                    adopted = true;
                    break;
                }
            }

            if !adopted {
                break;
            }
        }

        if current_cost < best_cost {
            best = current;
            best_cost = current_cost;
            best_moves = moves;
        }
    }

    Ok(TuneResult {
        kernel: best,
        cost_before,
        cost_after: best_cost,
        moves: best_moves,
        candidates_evaluated,
    })
}

fn tile_size_may_divide(level: &crate::primitives::LoopLevel, size: usize) -> bool {
    if size == 0 {
        return false;
    }
    let Some(lo) = constant_value(&level.lo) else {
        return false;
    };
    let Some(hi) = constant_value(&level.hi) else {
        return false;
    };
    let Ok(size) = i64::try_from(size) else {
        return false;
    };
    lo == 0 && hi >= 0 && hi % size == 0
}

fn random_rational_inputs(kernel: &Kernel, rng: &mut StdRng) -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    for buffer in kernel.buffers() {
        let BufferRole::Input(name) = buffer.role() else {
            continue;
        };
        let Some(numel) = buffer.shape().numel().ok() else {
            continue;
        };
        let values = (0..numel)
            .map(|_| rng.gen_range(-8..=8))
            .collect::<Vec<_>>();
        if let Ok(tensor) = Tensor::from_i64(buffer.shape().clone(), &values) {
            inputs.insert(name.clone(), tensor);
        }
    }
    inputs
}

fn random_f64_inputs(kernel: &Kernel, rng: &mut StdRng) -> HashMap<String, TensorF64> {
    let mut inputs = HashMap::new();
    for buffer in kernel.buffers() {
        let BufferRole::Input(name) = buffer.role() else {
            continue;
        };
        let Some(numel) = buffer.shape().numel().ok() else {
            continue;
        };
        let values = (0..numel)
            .map(|index| random_f64_value(index, rng))
            .collect::<Vec<_>>();
        if let Ok(tensor) = TensorF64::from_f64(buffer.shape().clone(), &values) {
            inputs.insert(name.clone(), tensor);
        }
    }
    inputs
}

fn random_f64_value(index: usize, rng: &mut StdRng) -> f64 {
    if index.is_multiple_of(13) {
        return -0.0;
    }
    if index.is_multiple_of(17) {
        return 0.0;
    }
    let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
    let mantissa = f64::from(rng.gen_range(1_u16..=63));
    let exponent = rng.gen_range(-4..=4);
    sign * mantissa * 2.0_f64.powi(exponent)
}
