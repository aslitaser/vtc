//! Mode-aware loop interchange.

use thiserror::Error;
use vtc_loopir::{Kernel, Stmt};

use crate::{DepError, LevelDep, Mode, classify_levels};

/// Errors produced by schedule legality checks.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LegalityError {
    /// The selected statement is not a perfect loop nest.
    #[error("statement is not a perfect loop nest")]
    NestNotPerfect,
    /// The selected nest or loop level is outside the kernel.
    #[error("loop level is out of range")]
    LevelOutOfRange,
    /// Strict mode cannot reorder a reduction loop.
    #[error("strict mode cannot reorder reduction loops")]
    ReductionReorderUnderStrict,
    /// The dependence classifier could not prove the swap legal.
    #[error("unknown dependence blocks interchange")]
    UnknownDependence,
    /// The selected loop bound was not a constant `0..N` bound.
    #[error("loop bound is not a constant zero-based range")]
    NonConstantBound,
    /// The requested tile size does not evenly divide the loop extent.
    #[error("tile size {tile_size} does not evenly divide extent {extent}")]
    NonDivisibleTile {
        /// Constant loop extent.
        extent: i64,
        /// Requested tile size.
        tile_size: usize,
    },
    /// Fusion requires adjacent top-level nests.
    #[error("fusion requires adjacent top-level nests")]
    FusionNotAdjacent,
    /// Fusion requires matching constant loop bounds.
    #[error("fusion loop bounds do not match")]
    FusionBoundsMismatch,
    /// Fusion would violate a producer/consumer or write/write dependence.
    #[error("fusion dependence violation")]
    FusionDependenceViolation,
    /// Fusion would violate an anti-dependence.
    #[error("fusion anti-dependence")]
    FusionAntiDependence,
    /// Internal consistency check failed.
    #[error("internal legality error: {0}")]
    Internal(String),
}

impl From<DepError> for LegalityError {
    fn from(error: DepError) -> Self {
        match error {
            DepError::NestNotPerfect => Self::NestNotPerfect,
        }
    }
}

/// Swaps adjacent loop levels in one top-level perfect nest.
///
/// The transform is structural and immutable: affine expressions are unchanged
/// because they reference loop variables by stable id. Termination and semantic
/// safety come from the mode-aware legality check; arbitrary unknown
/// dependences are refused.
///
/// # Errors
///
/// Returns [`LegalityError`] if the selected nest or levels are invalid, or if
/// the conservative dependence classifier cannot prove the requested swap legal
/// in `mode`.
pub fn interchange(
    kernel: &Kernel,
    nest: usize,
    level: usize,
    mode: Mode,
) -> Result<Kernel, LegalityError> {
    let stmt = kernel
        .body()
        .get(nest)
        .ok_or(LegalityError::LevelOutOfRange)?;
    let deps = classify_levels(stmt)?;
    let Some((&first, &second)) = deps.get(level).zip(deps.get(level.saturating_add(1))) else {
        return Err(LegalityError::LevelOutOfRange);
    };

    check_legality(first, second, mode)?;

    let mut body = kernel.body().to_vec();
    let Some(slot) = body.get_mut(nest) else {
        return Err(LegalityError::LevelOutOfRange);
    };
    *slot = swap_adjacent(stmt, level)?;

    Ok(Kernel::new_with_output_shapes(
        kernel.buffers().to_vec(),
        body,
        kernel.inputs().to_vec(),
        kernel.outputs().to_vec(),
        kernel.output_shapes().to_vec(),
    ))
}

fn check_legality(first: LevelDep, second: LevelDep, mode: Mode) -> Result<(), LegalityError> {
    match mode {
        Mode::Strict => {
            if matches!(first, LevelDep::Reduction) || matches!(second, LevelDep::Reduction) {
                Err(LegalityError::ReductionReorderUnderStrict)
            } else if matches!(first, LevelDep::Unknown) || matches!(second, LevelDep::Unknown) {
                Err(LegalityError::UnknownDependence)
            } else {
                Ok(())
            }
        }
        Mode::FastMath => {
            if matches!(first, LevelDep::Unknown) || matches!(second, LevelDep::Unknown) {
                Err(LegalityError::UnknownDependence)
            } else {
                Ok(())
            }
        }
    }
}

fn swap_adjacent(stmt: &Stmt, level: usize) -> Result<Stmt, LegalityError> {
    match stmt {
        Stmt::For { var, lo, hi, body } if level == 0 => {
            let [inner] = body.as_slice() else {
                return Err(LegalityError::NestNotPerfect);
            };
            let Stmt::For {
                var: inner_var,
                lo: inner_lo,
                hi: inner_hi,
                body: inner_body,
            } = inner
            else {
                return Err(LegalityError::NestNotPerfect);
            };
            Ok(Stmt::For {
                var: *inner_var,
                lo: inner_lo.clone(),
                hi: inner_hi.clone(),
                body: vec![Stmt::For {
                    var: *var,
                    lo: lo.clone(),
                    hi: hi.clone(),
                    body: inner_body.clone(),
                }],
            })
        }
        Stmt::For { var, lo, hi, body } => {
            let [inner] = body.as_slice() else {
                return Err(LegalityError::NestNotPerfect);
            };
            Ok(Stmt::For {
                var: *var,
                lo: lo.clone(),
                hi: hi.clone(),
                body: vec![swap_adjacent(inner, level.saturating_sub(1))?],
            })
        }
        Stmt::Assign { .. } => Err(LegalityError::NestNotPerfect),
    }
}
