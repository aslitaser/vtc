//! Schedule transforms and translation validation for `vtc`.
//!
//! This crate chooses and validates loop schedule transformations. It sits
//! above `vtc-loopir` and accepts only schedules that preserve loop semantics.

mod deps;
mod interchange;
mod mode;

pub use deps::{DepError, LevelDep, affine_depends_on, affine_eq, classify_levels};
pub use interchange::{LegalityError, interchange};
pub use mode::Mode;
