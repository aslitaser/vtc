//! Tensor dimension and shape types.

use std::fmt;

use crate::IrError;

/// A single tensor dimension extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Dim(usize);

impl Dim {
    /// Creates a dimension extent.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the dimension extent as a `usize`.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A tensor shape, stored as row-major logical dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<Dim>);

impl Shape {
    /// Creates a shape from dimensions in logical order.
    #[must_use]
    pub fn new(dims: Vec<Dim>) -> Self {
        Self(dims)
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the dimensions in logical order.
    #[must_use]
    pub fn dims(&self) -> &[Dim] {
        &self.0
    }

    /// Returns the number of logical elements in the shape.
    ///
    /// Rank-zero shapes contain one scalar element.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::ShapeOverflow`] if checked multiplication overflows.
    pub fn numel(&self) -> Result<usize, IrError> {
        self.0.iter().try_fold(1usize, |product, dim| {
            product.checked_mul(dim.get()).ok_or(IrError::ShapeOverflow)
        })
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (index, dim) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}
