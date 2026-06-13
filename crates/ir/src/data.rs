//! Literal tensor payloads for constants and inputs.

use crate::DType;

/// Row-major, logical tensor data.
///
/// This intentionally derives only `Debug` and `Clone`: floating-point buffers
/// make `Eq` and `Hash` unsound. Constant interning can be revisited later with
/// an explicit representation for float bit patterns.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// `f32` tensor payload.
    F32(Vec<f32>),
    /// `f64` tensor payload.
    F64(Vec<f64>),
    /// `i32` tensor payload.
    I32(Vec<i32>),
    /// `i64` tensor payload.
    I64(Vec<i64>),
    /// Boolean tensor payload.
    Bool(Vec<bool>),
}

impl TensorData {
    /// Returns the dtype represented by this payload.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Returns the number of logical elements stored in this payload.
    #[must_use]
    // Step 2 explicitly asks for `len()` as the TensorData payload-size API.
    #[allow(
        clippy::len_without_is_empty,
        reason = "step-2 public API requires len() without an is_empty() companion"
    )]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }
}
