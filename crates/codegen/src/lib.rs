//! Code generation backend for `vtc`.
//!
//! This crate lowers validated loop IR into target code. It sits above the
//! loop-IR boundary, where later verification claims stop and backend trust
//! begins.

mod bench;
mod emit_c;

pub use bench::{BenchError, MeasuredCost, compile_and_run, has_c_compiler, measure_runtime};
pub use emit_c::{CodegenError, emit_c};
