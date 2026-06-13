//! Reference interpreter and denotational semantics for `vtc`.
//!
//! This crate evaluates graph IR programs over exact rationals and over a
//! deterministic IEEE `f64` model. It sits above `vtc-ir` and provides the
//! executable oracles used by differential tests for rewrites and lowered loop
//! nests. The implementation is deliberately direct and unoptimized.

mod eval;
mod eval_f64;
mod tensor;
mod tensor_f64;

pub use eval::{EvalError, eval};
pub use eval_f64::eval_f64;
pub use tensor::Tensor;
pub use tensor_f64::TensorF64;
