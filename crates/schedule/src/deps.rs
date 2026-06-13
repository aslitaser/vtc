//! Conservative dependence classification for affine loop nests.

use std::collections::BTreeMap;

use thiserror::Error;
use vtc_loopir::{AffineExpr, BufferRef, LoopVar, ScalarExpr, Stmt};

/// Conservative dependence class for one loop level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelDep {
    /// Each iteration writes a distinct location and has no carried dependence.
    Parallel,
    /// The level carries a recognized associative self-accumulation.
    Reduction,
    /// The classifier cannot prove the level safe to reorder.
    Unknown,
}

/// Errors produced by dependence classification.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DepError {
    /// The statement is not a perfect loop nest ending in assignments.
    #[error("statement is not a perfect loop nest")]
    NestNotPerfect,
}

/// Returns true when two affine expressions are equal as functions.
///
/// Terms are compared by summed coefficient per loop variable, so absent
/// variables are equivalent to coefficient zero.
#[must_use]
pub fn affine_eq(left: &AffineExpr, right: &AffineExpr) -> bool {
    if left.constant_term() != right.constant_term() {
        return false;
    }
    coefficients(left) == coefficients(right)
}

/// Returns true when an affine expression has a nonzero coefficient for `var`.
#[must_use]
pub fn affine_depends_on(expr: &AffineExpr, var: LoopVar) -> bool {
    expr.terms()
        .iter()
        .filter(|(term_var, _)| *term_var == var)
        .fold(0i128, |acc, (_, coeff)| {
            acc.saturating_add(i128::from(*coeff))
        })
        != 0
}

/// Classifies every level in a perfect loop nest.
///
/// This analysis is sound but conservative and is tailored to the direct
/// lowering shapes in `vtc-loopir`: elementwise nests are classified as
/// parallel, and init-then-accumulate reductions are classified by recognizing
/// self-accumulation through `Add` or `Mul`. Anything outside those shapes is
/// reported as [`LevelDep::Unknown`] or rejected as non-perfect.
///
/// # Errors
///
/// Returns [`DepError::NestNotPerfect`] if `nest` is not a chain of `For`
/// statements ending in one or more assignments.
pub fn classify_levels(nest: &Stmt) -> Result<Vec<LevelDep>, DepError> {
    let (levels, body) = collect_perfect_nest(nest)?;
    if body.is_empty() {
        return Err(DepError::NestNotPerfect);
    }

    levels
        .iter()
        .copied()
        .map(|level| classify_level(level, body))
        .collect()
}

fn coefficients(expr: &AffineExpr) -> BTreeMap<LoopVar, i128> {
    let mut out: BTreeMap<LoopVar, i128> = BTreeMap::new();
    for (var, coeff) in expr.terms() {
        let entry = out.entry(*var).or_insert(0);
        *entry = (*entry).saturating_add(i128::from(*coeff));
    }
    out.retain(|_, coeff| *coeff != 0);
    out
}

fn collect_perfect_nest(root: &Stmt) -> Result<(Vec<LoopVar>, &[Stmt]), DepError> {
    let mut levels = Vec::new();
    let mut current = root;

    loop {
        match current {
            Stmt::For { var, body, .. } => {
                levels.push(*var);
                if body.len() == 1 && matches!(body.first(), Some(Stmt::For { .. })) {
                    let Some(child) = body.first() else {
                        return Err(DepError::NestNotPerfect);
                    };
                    current = child;
                } else if body.iter().all(|stmt| matches!(stmt, Stmt::Assign { .. })) {
                    return Ok((levels, body));
                } else {
                    return Err(DepError::NestNotPerfect);
                }
            }
            Stmt::Assign { .. } => return Err(DepError::NestNotPerfect),
        }
    }
}

fn classify_level(level: LoopVar, body: &[Stmt]) -> Result<LevelDep, DepError> {
    let assignments = body
        .iter()
        .map(|stmt| match stmt {
            Stmt::Assign { target, value } => Ok((target, value)),
            Stmt::For { .. } => Err(DepError::NestNotPerfect),
        })
        .collect::<Result<Vec<_>, _>>()?;

    if assignments
        .iter()
        .any(|(target, value)| is_self_accumulation_carried_by(target, value, level))
    {
        return Ok(LevelDep::Reduction);
    }

    if assignments
        .iter()
        .all(|(target, _)| affine_depends_on(&target.index, level))
    {
        Ok(LevelDep::Parallel)
    } else {
        Ok(LevelDep::Unknown)
    }
}

fn is_self_accumulation_carried_by(target: &BufferRef, value: &ScalarExpr, level: LoopVar) -> bool {
    if affine_depends_on(&target.index, level) {
        return false;
    }
    combiner_self_load(value)
        .is_some_and(|load| load.buffer == target.buffer && affine_eq(&load.index, &target.index))
}

fn combiner_self_load(value: &ScalarExpr) -> Option<&BufferRef> {
    match value {
        ScalarExpr::Add(left, right) | ScalarExpr::Mul(left, right) => {
            load_ref(left).or_else(|| load_ref(right))
        }
        ScalarExpr::Load(_)
        | ScalarExpr::ConstScalar(_)
        | ScalarExpr::Sub(_, _)
        | ScalarExpr::Neg(_)
        | ScalarExpr::Relu(_) => None,
    }
}

fn load_ref(value: &ScalarExpr) -> Option<&BufferRef> {
    match value {
        ScalarExpr::Load(reference) => Some(reference),
        ScalarExpr::ConstScalar(_)
        | ScalarExpr::Add(_, _)
        | ScalarExpr::Sub(_, _)
        | ScalarExpr::Mul(_, _)
        | ScalarExpr::Neg(_)
        | ScalarExpr::Relu(_) => None,
    }
}
