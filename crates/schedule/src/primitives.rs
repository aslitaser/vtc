//! Shared structural helpers for schedule primitives.

use std::collections::{BTreeMap, BTreeSet};

use vtc_loopir::{AffineExpr, BufferId, BufferRef, Kernel, LoopVar, ScalarExpr, Stmt};

use crate::LegalityError;

/// One loop wrapper in a perfect nest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopLevel {
    /// Induction variable.
    pub(crate) var: LoopVar,
    /// Inclusive lower bound.
    pub(crate) lo: AffineExpr,
    /// Exclusive upper bound.
    pub(crate) hi: AffineExpr,
}

/// A decomposed perfect loop nest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PerfectNest {
    /// Loop wrappers from outermost to innermost.
    pub(crate) levels: Vec<LoopLevel>,
    /// Innermost statements.
    pub(crate) body: Vec<Stmt>,
}

/// Read and write buffer accesses under a statement list.
#[derive(Debug, Clone, Default)]
pub(crate) struct AccessSummary {
    /// Buffer reads.
    pub(crate) reads: Vec<BufferRef>,
    /// Buffer writes.
    pub(crate) writes: Vec<BufferRef>,
}

/// Decomposes a statement into a perfect loop nest.
pub(crate) fn decompose_perfect_nest(stmt: &Stmt) -> Result<PerfectNest, LegalityError> {
    let mut levels = Vec::new();
    let mut current = stmt;

    loop {
        match current {
            Stmt::For { var, lo, hi, body } => {
                levels.push(LoopLevel {
                    var: *var,
                    lo: lo.clone(),
                    hi: hi.clone(),
                });
                if body.len() == 1 && matches!(body.first(), Some(Stmt::For { .. })) {
                    let Some(child) = body.first() else {
                        return Err(LegalityError::NestNotPerfect);
                    };
                    current = child;
                } else if body
                    .iter()
                    .all(|inner| matches!(inner, Stmt::Assign { .. }))
                {
                    return Ok(PerfectNest {
                        levels,
                        body: body.clone(),
                    });
                } else {
                    return Err(LegalityError::NestNotPerfect);
                }
            }
            Stmt::Assign { .. } => return Err(LegalityError::NestNotPerfect),
        }
    }
}

/// Rebuilds a perfect loop nest from levels and an innermost body.
pub(crate) fn build_perfect_nest(
    levels: &[LoopLevel],
    body: Vec<Stmt>,
) -> Result<Stmt, LegalityError> {
    if levels.is_empty() {
        return Err(LegalityError::Internal(
            "cannot build an empty loop nest".to_owned(),
        ));
    }
    let mut current = body;
    for level in levels.iter().rev() {
        current = vec![Stmt::For {
            var: level.var,
            lo: level.lo.clone(),
            hi: level.hi.clone(),
            body: current,
        }];
    }
    current
        .into_iter()
        .next()
        .ok_or_else(|| LegalityError::Internal("rebuilt loop nest unexpectedly empty".to_owned()))
}

/// Returns a constant expression value when no loop variable is present.
pub(crate) fn constant_value(expr: &AffineExpr) -> Option<i64> {
    if expr.terms().iter().any(|(_, coeff)| *coeff != 0) {
        None
    } else {
        Some(expr.constant_term())
    }
}

/// Returns fresh loop variables that do not collide with any variable in the kernel.
pub(crate) fn fresh_loop_vars(
    kernel: &Kernel,
    count: usize,
) -> Result<Vec<LoopVar>, LegalityError> {
    let mut vars = BTreeSet::new();
    for stmt in kernel.body() {
        collect_stmt_vars(stmt, &mut vars);
    }

    let start = vars
        .iter()
        .next_back()
        .map_or(0, |var| var.id().saturating_add(1));
    (0..count)
        .map(|offset| {
            let offset = u32::try_from(offset)
                .map_err(|_| LegalityError::Internal("fresh loop variable overflow".to_owned()))?;
            let id = start.checked_add(offset).ok_or_else(|| {
                LegalityError::Internal("fresh loop variable overflow".to_owned())
            })?;
            Ok(LoopVar::new(id))
        })
        .collect()
}

/// Substitutes one loop variable throughout a statement list.
pub(crate) fn substitute_stmts(
    stmts: &[Stmt],
    var: LoopVar,
    replacement: &AffineExpr,
) -> Vec<Stmt> {
    stmts
        .iter()
        .map(|stmt| substitute_stmt(stmt, var, replacement))
        .collect()
}

/// Substitutes a set of loop variables throughout a statement list.
pub(crate) fn substitute_many_stmts(
    stmts: &[Stmt],
    replacements: &[(LoopVar, AffineExpr)],
) -> Vec<Stmt> {
    replacements
        .iter()
        .fold(stmts.to_vec(), |current, (var, replacement)| {
            substitute_stmts(&current, *var, replacement)
        })
}

/// Collects all buffer reads and writes from a statement list.
pub(crate) fn summarize_accesses(stmts: &[Stmt]) -> AccessSummary {
    let mut summary = AccessSummary::default();
    for stmt in stmts {
        summarize_stmt(stmt, &mut summary);
    }
    summary
}

/// Builds a map from written buffer id to every written index.
pub(crate) fn writes_by_buffer(writes: &[BufferRef]) -> BTreeMap<BufferId, Vec<AffineExpr>> {
    let mut out: BTreeMap<BufferId, Vec<AffineExpr>> = BTreeMap::new();
    for write in writes {
        out.entry(write.buffer)
            .or_default()
            .push(write.index.clone());
    }
    out
}

fn substitute_stmt(stmt: &Stmt, var: LoopVar, replacement: &AffineExpr) -> Stmt {
    match stmt {
        Stmt::For {
            var: loop_var,
            lo,
            hi,
            body,
        } => Stmt::For {
            var: *loop_var,
            lo: lo.substitute(var, replacement),
            hi: hi.substitute(var, replacement),
            body: substitute_stmts(body, var, replacement),
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: substitute_ref(target, var, replacement),
            value: substitute_scalar(value, var, replacement),
        },
    }
}

fn substitute_ref(reference: &BufferRef, var: LoopVar, replacement: &AffineExpr) -> BufferRef {
    BufferRef {
        buffer: reference.buffer,
        index: reference.index.substitute(var, replacement),
    }
}

fn substitute_scalar(value: &ScalarExpr, var: LoopVar, replacement: &AffineExpr) -> ScalarExpr {
    match value {
        ScalarExpr::Load(reference) => {
            ScalarExpr::Load(substitute_ref(reference, var, replacement))
        }
        ScalarExpr::ConstScalar(value) => ScalarExpr::ConstScalar(value.clone()),
        ScalarExpr::Add(left, right) => ScalarExpr::Add(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Sub(left, right) => ScalarExpr::Sub(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Mul(left, right) => ScalarExpr::Mul(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Neg(input) => {
            ScalarExpr::Neg(Box::new(substitute_scalar(input, var, replacement)))
        }
        ScalarExpr::Relu(input) => {
            ScalarExpr::Relu(Box::new(substitute_scalar(input, var, replacement)))
        }
    }
}

fn summarize_stmt(stmt: &Stmt, summary: &mut AccessSummary) {
    match stmt {
        Stmt::For { body, .. } => {
            for inner in body {
                summarize_stmt(inner, summary);
            }
        }
        Stmt::Assign { target, value } => {
            summary.writes.push(target.clone());
            summarize_scalar(value, summary);
        }
    }
}

fn summarize_scalar(value: &ScalarExpr, summary: &mut AccessSummary) {
    match value {
        ScalarExpr::Load(reference) => summary.reads.push(reference.clone()),
        ScalarExpr::ConstScalar(_) => {}
        ScalarExpr::Add(left, right)
        | ScalarExpr::Sub(left, right)
        | ScalarExpr::Mul(left, right) => {
            summarize_scalar(left, summary);
            summarize_scalar(right, summary);
        }
        ScalarExpr::Neg(input) | ScalarExpr::Relu(input) => summarize_scalar(input, summary),
    }
}

fn collect_stmt_vars(stmt: &Stmt, vars: &mut BTreeSet<LoopVar>) {
    match stmt {
        Stmt::For { var, lo, hi, body } => {
            vars.insert(*var);
            collect_expr_vars(lo, vars);
            collect_expr_vars(hi, vars);
            for inner in body {
                collect_stmt_vars(inner, vars);
            }
        }
        Stmt::Assign { target, value } => {
            collect_expr_vars(&target.index, vars);
            collect_scalar_vars(value, vars);
        }
    }
}

fn collect_scalar_vars(value: &ScalarExpr, vars: &mut BTreeSet<LoopVar>) {
    match value {
        ScalarExpr::Load(reference) => collect_expr_vars(&reference.index, vars),
        ScalarExpr::ConstScalar(_) => {}
        ScalarExpr::Add(left, right)
        | ScalarExpr::Sub(left, right)
        | ScalarExpr::Mul(left, right) => {
            collect_scalar_vars(left, vars);
            collect_scalar_vars(right, vars);
        }
        ScalarExpr::Neg(input) | ScalarExpr::Relu(input) => collect_scalar_vars(input, vars),
    }
}

fn collect_expr_vars(expr: &AffineExpr, vars: &mut BTreeSet<LoopVar>) {
    for (var, coeff) in expr.terms() {
        if *coeff != 0 {
            vars.insert(*var);
        }
    }
}
