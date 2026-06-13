//! Adjacent elementwise loop fusion.

use std::collections::BTreeSet;

use vtc_loopir::{AffineExpr, BufferId, BufferRef, Kernel};

use crate::primitives::{
    LoopLevel, build_perfect_nest, constant_value, decompose_perfect_nest, fresh_loop_vars,
    substitute_many_stmts, summarize_accesses, writes_by_buffer,
};
use crate::{LegalityError, Mode, affine_eq};

/// Merges two adjacent top-level perfect loop nests.
///
/// Fusion is legal when the nests have identical constant bounds, the consumer
/// reads each producer-written buffer at the same iteration index, anti-
/// dependences are absent, and the two nests write disjoint buffers. The
/// intermediate buffer is retained; scalar replacement is a later optimization.
///
/// # Errors
///
/// Returns [`LegalityError`] if the nests are not adjacent, are not perfect,
/// have mismatched bounds, or violate the fusion dependence checks.
pub fn fuse(
    kernel: &Kernel,
    nest_a: usize,
    nest_b: usize,
    _mode: Mode,
) -> Result<Kernel, LegalityError> {
    if Some(nest_b) != nest_a.checked_add(1) {
        return Err(LegalityError::FusionNotAdjacent);
    }

    let first_stmt = kernel
        .body()
        .get(nest_a)
        .ok_or(LegalityError::LevelOutOfRange)?;
    let second_stmt = kernel
        .body()
        .get(nest_b)
        .ok_or(LegalityError::LevelOutOfRange)?;
    let first = decompose_perfect_nest(first_stmt)?;
    let second = decompose_perfect_nest(second_stmt)?;
    ensure_bounds_match(&first.levels, &second.levels)?;

    let shared_vars = fresh_loop_vars(kernel, first.levels.len())?;
    let first_replacements = first
        .levels
        .iter()
        .zip(&shared_vars)
        .map(|(level, var)| (level.var, AffineExpr::var(*var)))
        .collect::<Vec<_>>();
    let second_replacements = second
        .levels
        .iter()
        .zip(&shared_vars)
        .map(|(level, var)| (level.var, AffineExpr::var(*var)))
        .collect::<Vec<_>>();

    let first_body = substitute_many_stmts(&first.body, &first_replacements);
    let second_body = substitute_many_stmts(&second.body, &second_replacements);
    check_fusion_legality(&first_body, &second_body)?;

    let shared_levels = first
        .levels
        .iter()
        .zip(shared_vars)
        .map(|(level, var)| LoopLevel {
            var,
            lo: level.lo.clone(),
            hi: level.hi.clone(),
        })
        .collect::<Vec<_>>();
    let mut fused_body = first_body;
    fused_body.extend(second_body);
    let fused = build_perfect_nest(&shared_levels, fused_body)?;

    let mut body = Vec::with_capacity(kernel.body().len().saturating_sub(1));
    for (index, stmt) in kernel.body().iter().enumerate() {
        if index == nest_a {
            body.push(fused.clone());
        } else if index != nest_b {
            body.push(stmt.clone());
        }
    }

    Ok(Kernel::new_with_output_shapes(
        kernel.buffers().to_vec(),
        body,
        kernel.inputs().to_vec(),
        kernel.outputs().to_vec(),
        kernel.output_shapes().to_vec(),
    ))
}

fn ensure_bounds_match(first: &[LoopLevel], second: &[LoopLevel]) -> Result<(), LegalityError> {
    if first.len() != second.len() {
        return Err(LegalityError::FusionBoundsMismatch);
    }
    for (left, right) in first.iter().zip(second) {
        if constant_value(&left.lo).is_none()
            || constant_value(&left.hi).is_none()
            || constant_value(&right.lo).is_none()
            || constant_value(&right.hi).is_none()
            || !affine_eq(&left.lo, &right.lo)
            || !affine_eq(&left.hi, &right.hi)
        {
            return Err(LegalityError::FusionBoundsMismatch);
        }
    }
    Ok(())
}

fn check_fusion_legality(
    first_body: &[vtc_loopir::Stmt],
    second_body: &[vtc_loopir::Stmt],
) -> Result<(), LegalityError> {
    let first_accesses = summarize_accesses(first_body);
    let second_accesses = summarize_accesses(second_body);
    let first_writes = writes_by_buffer(&first_accesses.writes);
    let second_writes = writes_by_buffer(&second_accesses.writes);

    if !buffer_set(first_writes.keys().copied())
        .is_disjoint(&buffer_set(second_writes.keys().copied()))
    {
        return Err(LegalityError::FusionDependenceViolation);
    }

    if !buffer_set(
        first_accesses
            .reads
            .iter()
            .map(|reference| reference.buffer),
    )
    .is_disjoint(&buffer_set(second_writes.keys().copied()))
    {
        return Err(LegalityError::FusionAntiDependence);
    }

    let first_read_write = buffer_set(
        first_accesses
            .reads
            .iter()
            .map(|reference| reference.buffer),
    )
    .intersection(&buffer_set(first_writes.keys().copied()))
    .copied()
    .collect::<BTreeSet<_>>();

    for read in &second_accesses.reads {
        let Some(indices) = first_writes.get(&read.buffer) else {
            continue;
        };
        if first_read_write.contains(&read.buffer) || !has_single_aligned_write(indices, read) {
            return Err(LegalityError::FusionDependenceViolation);
        }
    }

    Ok(())
}

fn has_single_aligned_write(indices: &[AffineExpr], read: &BufferRef) -> bool {
    let [index] = indices else {
        return false;
    };
    affine_eq(index, &read.index)
}

fn buffer_set(buffers: impl IntoIterator<Item = BufferId>) -> BTreeSet<BufferId> {
    buffers.into_iter().collect()
}
