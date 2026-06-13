//! Reference interpreter and denotational semantics for `vtc`.
//!
//! This crate evaluates graph IR programs over exact rationals. It sits above
//! `vtc-ir` and provides the executable oracle used by future differential
//! tests for rewrites and lowered loop nests. The implementation is deliberately
//! direct and unoptimized.

mod eval;
mod tensor;

pub use eval::{EvalError, eval};
pub use tensor::Tensor;
