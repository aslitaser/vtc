//! Schedule transforms and translation validation for `vtc`.
//!
//! This crate chooses and validates loop schedule transformations. It sits
//! above `vtc-loopir` and accepts only schedules that preserve loop semantics.

mod autotune;
mod cost;
mod deps;
mod fuse;
mod interchange;
mod mode;
mod primitives;
mod tile;

pub use autotune::{
    MoveDesc, TuneConfig, TuneError, TuneResult, autotune, legal_moves, validate_equiv,
};
pub use cost::StaticCost;
pub use deps::{DepError, LevelDep, affine_depends_on, affine_eq, classify_levels};
pub use fuse::fuse;
pub use interchange::{LegalityError, interchange};
pub use mode::Mode;
pub use tile::tile;
pub use vtc_loopir::CostModel;
