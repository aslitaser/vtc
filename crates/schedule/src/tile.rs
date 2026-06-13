//! Order-preserving strip-mining.

use vtc_loopir::{AffineExpr, Kernel};

use crate::primitives::{
    LoopLevel, build_perfect_nest, constant_value, decompose_perfect_nest, fresh_loop_vars,
    substitute_stmts,
};
use crate::{LegalityError, Mode};

/// Strip-mines one loop level of a top-level perfect nest.
///
/// This transform replaces `for i in 0..N` with `for io in 0..N/T { for ii
/// in 0..T { ... } }` and substitutes every use of `i` with `T*io + ii`.
/// It preserves ascending execution order and therefore is legal in both
/// modes. Cache blocking is expressed by composing tile with interchange; the
/// mode-aware reordering rules live in `interchange`.
///
/// # Errors
///
/// Returns [`LegalityError`] when the selected nest or level is invalid, the
/// loop bound is not a constant `0..N`, or `tile_size` does not divide `N`.
pub fn tile(
    kernel: &Kernel,
    nest: usize,
    level: usize,
    tile_size: usize,
    _mode: Mode,
) -> Result<Kernel, LegalityError> {
    let stmt = kernel
        .body()
        .get(nest)
        .ok_or(LegalityError::LevelOutOfRange)?;
    let decomposed = decompose_perfect_nest(stmt)?;
    let selected = decomposed
        .levels
        .get(level)
        .ok_or(LegalityError::LevelOutOfRange)?;
    let extent = zero_based_extent(selected)?;
    let tile_i64 = i64::try_from(tile_size)
        .map_err(|_| LegalityError::NonDivisibleTile { extent, tile_size })?;
    if tile_size == 0 || extent % tile_i64 != 0 {
        return Err(LegalityError::NonDivisibleTile { extent, tile_size });
    }

    let fresh = fresh_loop_vars(kernel, 2)?;
    let [outer, inner] = fresh.as_slice() else {
        return Err(LegalityError::Internal(
            "fresh variable allocator returned wrong count".to_owned(),
        ));
    };
    let replacement = AffineExpr::new(vec![(*outer, tile_i64), (*inner, 1)], 0);
    let mut levels = Vec::new();
    levels.extend_from_slice(&decomposed.levels[..level]);
    levels.push(LoopLevel {
        var: *outer,
        lo: AffineExpr::constant(0),
        hi: AffineExpr::constant(extent / tile_i64),
    });
    levels.push(LoopLevel {
        var: *inner,
        lo: AffineExpr::constant(0),
        hi: AffineExpr::constant(tile_i64),
    });
    for suffix in &decomposed.levels[level.saturating_add(1)..] {
        levels.push(LoopLevel {
            var: suffix.var,
            lo: suffix.lo.substitute(selected.var, &replacement),
            hi: suffix.hi.substitute(selected.var, &replacement),
        });
    }

    let inner_body = substitute_stmts(&decomposed.body, selected.var, &replacement);
    let mut body = kernel.body().to_vec();
    let Some(slot) = body.get_mut(nest) else {
        return Err(LegalityError::LevelOutOfRange);
    };
    *slot = build_perfect_nest(&levels, inner_body)?;

    Ok(Kernel::new_with_output_shapes(
        kernel.buffers().to_vec(),
        body,
        kernel.inputs().to_vec(),
        kernel.outputs().to_vec(),
        kernel.output_shapes().to_vec(),
    ))
}

fn zero_based_extent(level: &LoopLevel) -> Result<i64, LegalityError> {
    let Some(lo) = constant_value(&level.lo) else {
        return Err(LegalityError::NonConstantBound);
    };
    let Some(hi) = constant_value(&level.hi) else {
        return Err(LegalityError::NonConstantBound);
    };
    if lo == 0 && hi >= 0 {
        Ok(hi)
    } else {
        Err(LegalityError::NonConstantBound)
    }
}
