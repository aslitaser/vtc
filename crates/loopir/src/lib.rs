//! Affine loop IR and lowering boundary for `vtc`.
//!
//! This crate holds the loop-level representation produced from graph IR. It
//! sits above the graph semantics layer and below scheduling and codegen.

mod interp_loops;
mod interp_loops_f64;
mod loop_ir;
mod lower;

pub use interp_loops::{LoopError, eval_loops};
pub use interp_loops_f64::eval_loops_f64;
pub use loop_ir::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopVar, ScalarExpr, Stmt,
};
pub use lower::{LowerError, lower};
pub use vtc_interp::TensorF64;
