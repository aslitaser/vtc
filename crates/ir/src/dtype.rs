//! Scalar element types for graph tensors.

use std::fmt;

/// Scalar element type carried by a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE single-precision floating-point storage.
    F32,
    /// IEEE double-precision floating-point storage.
    F64,
    /// Signed 32-bit integer storage.
    I32,
    /// Signed 64-bit integer storage.
    I64,
    /// Boolean storage.
    Bool,
}

impl DType {
    /// Returns whether this dtype is a floating-point storage type.
    ///
    /// The reference semantics introduced later is exact-rational; this method
    /// only classifies the storage dtype used by graph IR nodes.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns whether this dtype supports numeric tensor operations.
    #[must_use]
    pub const fn is_numeric(&self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::I32 | Self::I64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
        };
        f.write_str(text)
    }
}
