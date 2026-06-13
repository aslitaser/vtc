//! Graph IR types for `vtc`.
//!
//! This crate is the bottom layer of the compiler workspace. It defines the
//! graph-level tensor program representation that every higher layer reads or
//! transforms. The crate owns structural data, structural validation, and
//! pure type inference. Interpretation, rewriting, scheduling, and code
//! generation live in higher layers.

// The module names intentionally mirror the design-request file layout, which
// makes names such as `TensorType` in `tensor_type.rs` clearer than shorter
// alternatives.
#![allow(
    clippy::module_name_repetitions,
    reason = "module names mirror the step-2 file layout"
)]

mod builder;
mod data;
mod dtype;
mod error;
mod graph;
mod infer;
mod op;
mod shape;
mod tensor_type;

pub use builder::GraphBuilder;
pub use data::TensorData;
pub use dtype::DType;
pub use error::IrError;
pub use graph::{Graph, Node, NodeId};
pub use infer::{GraphTypes, TypeError, infer_types};
pub use op::Op;
pub use shape::{Dim, Shape};
pub use tensor_type::TensorType;
