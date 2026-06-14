//! Command-line pipeline wiring for `vtc`.
//!
//! The CLI crate is the top layer of the workspace. It does not define new
//! compiler logic; it composes the graph, rewrite, lowering, scheduling, and
//! code generation crates behind a small built-in example frontend.

mod examples;
mod pipeline;

pub use examples::{build_example, list_examples};
pub use pipeline::{
    CheckStatus, CostKind, PipelineError, RunOpts, RunReport, RunStatus, parse_shape_spec,
    run_pipeline,
};
