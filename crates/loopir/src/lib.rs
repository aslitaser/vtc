//! Affine loop IR and lowering boundary for `vtc`.
//!
//! This crate holds the loop-level representation produced from graph IR. It
//! sits above the graph semantics layer and below scheduling and codegen.

mod interp_loops;
mod interp_loops_f64;
mod loop_ir;
mod lower;

/// Cost model over lowered loop kernels.
///
/// The schedule autotuner and backend benchmarking use this shared trait so
/// static estimates and measured runtimes can be plugged into the same search.
pub trait CostModel {
    /// Returns a deterministic cost estimate for `kernel`.
    fn cost(&self, kernel: &Kernel) -> u64;
}

pub use interp_loops::{LoopError, eval_loops};
pub use interp_loops_f64::eval_loops_f64;
pub use loop_ir::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopVar, ScalarExpr, Stmt,
};
pub use lower::{LowerError, lower};
pub use vtc_interp::TensorF64;
