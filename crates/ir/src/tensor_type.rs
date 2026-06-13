//! Tensor type metadata for graph IR values.

use crate::{DType, Shape};

/// Tensor element dtype and logical shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    dtype: DType,
    shape: Shape,
}

impl TensorType {
    /// Creates a tensor type.
    #[must_use]
    pub const fn new(dtype: DType, shape: Shape) -> Self {
        Self { dtype, shape }
    }

    /// Returns the tensor element dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }
}
