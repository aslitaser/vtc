//! Heuristic cost models for schedule search.

use std::collections::{BTreeMap, BTreeSet};

use vtc_loopir::{AffineExpr, BufferId, BufferRole, Kernel, LoopVar, Stmt};

use crate::primitives::{constant_value, decompose_perfect_nest, summarize_accesses};

const TEMP_BUFFER_PENALTY: u64 = 16;
const TEMP_BOUNDARY_PENALTY: u64 = 128;
const NEST_OVERHEAD: u64 = 8;

/// Cost model used by the autotuner.
///
/// Implementations are allowed to be heuristic. Lower values are considered
/// better by the current greedy search.
pub trait CostModel {
    /// Returns a deterministic cost estimate for `kernel`.
    fn cost(&self, kernel: &Kernel) -> u64;
}

/// Static deterministic heuristic cost proxy.
///
/// This is not a runtime measurement. It estimates loop traffic from trip
/// counts and distinct buffer accesses, rewards unit-stride innermost access,
/// applies a small credit for strip-mined nests, and penalizes temporary
/// buffers that must be materialized across top-level nest boundaries. A
/// measured-runtime model can be plugged in later through [`CostModel`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticCost;

impl CostModel for StaticCost {
    fn cost(&self, kernel: &Kernel) -> u64 {
        let loop_cost = kernel
            .body()
            .iter()
            .map(stmt_cost)
            .fold(0u64, u64::saturating_add);
        loop_cost
            .saturating_add(temp_buffer_cost(kernel))
            .saturating_add(temp_boundary_cost(kernel))
    }
}

fn stmt_cost(stmt: &Stmt) -> u64 {
    let Ok(nest) = decompose_perfect_nest(stmt) else {
        return NEST_OVERHEAD;
    };
    let trips = nest
        .levels
        .iter()
        .map(|level| constant_extent(&level.lo, &level.hi).unwrap_or(1))
        .fold(1u64, u64::saturating_mul);
    let inner_var = nest.levels.last().map(|level| level.var);
    let accesses = summarize_accesses(&nest.body);
    let mut distinct = BTreeSet::new();
    let access_weight = accesses
        .reads
        .iter()
        .chain(&accesses.writes)
        .filter(|reference| distinct.insert((reference.buffer, reference.index.to_string())))
        .map(|reference| access_cost(&reference.index, inner_var))
        .fold(0u64, u64::saturating_add)
        .max(1);

    let raw = trips
        .saturating_mul(access_weight)
        .saturating_add(NEST_OVERHEAD);
    raw / strip_mine_credit(&nest.levels).max(1)
}

fn constant_extent(lo: &AffineExpr, hi: &AffineExpr) -> Option<u64> {
    let lo = constant_value(lo)?;
    let hi = constant_value(hi)?;
    if lo > hi {
        None
    } else {
        u64::try_from(hi.saturating_sub(lo)).ok()
    }
}

fn access_cost(index: &AffineExpr, inner_var: Option<LoopVar>) -> u64 {
    let Some(inner_var) = inner_var else {
        return 2;
    };
    let coeff = coeff_for(index, inner_var).unsigned_abs();
    match coeff {
        1 => 1,
        0 => 2,
        2..=4 => 3,
        _ => 5,
    }
}

fn strip_mine_credit(levels: &[crate::primitives::LoopLevel]) -> u64 {
    if levels.len() <= 3 {
        return 1;
    }
    levels
        .iter()
        .filter_map(|level| constant_extent(&level.lo, &level.hi))
        .filter(|extent| (2..=32).contains(extent))
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn coeff_for(expr: &AffineExpr, var: LoopVar) -> i64 {
    expr.terms()
        .iter()
        .filter_map(|(term_var, coeff)| (*term_var == var).then_some(*coeff))
        .fold(0i64, i64::saturating_add)
}

fn temp_buffer_cost(kernel: &Kernel) -> u64 {
    let temps = kernel
        .buffers()
        .iter()
        .filter(|buffer| matches!(buffer.role(), BufferRole::Temp))
        .count();
    u64::try_from(temps)
        .unwrap_or(u64::MAX)
        .saturating_mul(TEMP_BUFFER_PENALTY)
}

fn temp_boundary_cost(kernel: &Kernel) -> u64 {
    let temp_buffers = kernel
        .buffers()
        .iter()
        .filter_map(|buffer| matches!(buffer.role(), BufferRole::Temp).then_some(buffer.id()))
        .collect::<BTreeSet<_>>();
    let mut writes: BTreeMap<BufferId, usize> = BTreeMap::new();
    let mut reads: BTreeMap<BufferId, usize> = BTreeMap::new();

    for (index, stmt) in kernel.body().iter().enumerate() {
        let Ok(nest) = decompose_perfect_nest(stmt) else {
            continue;
        };
        let accesses = summarize_accesses(&nest.body);
        for write in accesses.writes {
            if temp_buffers.contains(&write.buffer) {
                writes.entry(write.buffer).or_insert(index);
            }
        }
        for read in accesses.reads {
            if temp_buffers.contains(&read.buffer) {
                reads.entry(read.buffer).or_insert(index);
            }
        }
    }

    let boundaries = writes
        .iter()
        .filter(|(buffer, write_index)| {
            reads
                .get(buffer)
                .is_some_and(|read_index| read_index > *write_index)
        })
        .count();
    u64::try_from(boundaries)
        .unwrap_or(u64::MAX)
        .saturating_mul(TEMP_BOUNDARY_PENALTY)
}
