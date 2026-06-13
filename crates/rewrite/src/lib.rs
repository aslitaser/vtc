//! Graph rewrite rules and rewrite engine for `vtc`.
//!
//! This crate hosts the scaffolding for semantics-preserving graph rewrites. It
//! sits above `vtc-ir` and records each rewrite's numeric safety from explicit
//! algebraic laws, graph surgery helpers, and differential-test generators.

pub mod r#gen;

mod driver;
mod rule;
mod rules;
mod ruleset;
mod safety;
mod surgery;

pub use driver::{AppliedRewrite, DriverConfig, RunResult, run};
pub use rule::Rewrite;
pub use rules::{NegNegElim, ReluIdempotentElim, ReshapeReshapeFuse};
pub use ruleset::RuleSet;
pub use safety::{Law, NumericSafety, RewriteMode};
pub use surgery::{SurgeryError, prune_and_rebuild, redefine_node, replace_all_uses};
