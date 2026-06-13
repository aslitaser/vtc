//! Graph rewrite rules and rewrite engine for `vtc`.
//!
//! This crate hosts the scaffolding for semantics-preserving graph rewrites. It
//! sits above `vtc-ir` and records each rewrite's numeric safety from explicit
//! algebraic laws, graph surgery helpers, and differential-test generators.

pub mod r#gen;

mod driver;
mod egg_lang;
mod egg_search;
mod rule;
mod rules;
mod ruleset;
mod safety;
mod surgery;

pub use driver::{AppliedRewrite, DriverConfig, RunResult, run};
pub use egg_lang::{AtomTable, EggError, EggLang, graph_to_recexpr, recexpr_to_graph};
pub use egg_search::{
    EggConfig, EggResult, EggRule, egg_rules_for_mode, make_egg_rule, optimize_with_egg,
    optimize_with_egg_rules,
};
pub use rule::Rewrite;
pub use rules::{NegNegElim, ReluIdempotentElim, ReshapeReshapeFuse};
pub use ruleset::RuleSet;
pub use safety::{Law, NumericSafety, RewriteMode};
pub use surgery::{SurgeryError, prune_and_rebuild, redefine_node, replace_all_uses};
