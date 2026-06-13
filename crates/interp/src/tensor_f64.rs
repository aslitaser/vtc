//! Runtime tensor values for the IEEE `f64` interpreter.

use vtc_ir::{DType, Shape, TensorData};

use crate::EvalError;
use crate::tensor::shape_numel;

/// Runtime tensor value with row-major IEEE `f64` data.
#[derive(Debug, Clone)]
pub struct TensorF64 {
    shape: Shape,
    data: Vec<f64>,
}

impl TensorF64 {
    /// Creates a tensor from a shape and row-major `f64` payload.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DataShapeMismatch`] if the payload length does not
    /// match the shape element count.
    pub fn new(shape: Shape, data: Vec<f64>) -> Result<Self, EvalError> {
        let expected = shape_numel(&shape)?;
        let got = data.len();
        if expected != got {
            return Err(EvalError::DataShapeMismatch { expected, got });
        }

        Ok(Self { shape, data })
    }

    /// Lifts nominal tensor data into the IEEE `f64` interpreter domain.
    ///
    /// Integer payloads are converted with Rust's integer-to-`f64` cast, which
    /// is exact only up to 2^53. `f32` payloads are widened exactly. Floating
    /// constants must be finite, and booleans are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] if the payload length mismatches the shape, a
    /// floating-point constant is not finite, or the dtype is unsupported.
    pub fn from_data(data: &TensorData, shape: &Shape) -> Result<Self, EvalError> {
        let lifted = match data {
            TensorData::F32(values) => values
                .iter()
                .map(|&value| finite_f64(f64::from(value)))
                .collect::<Result<Vec<_>, _>>()?,
            TensorData::F64(values) => values
                .iter()
                .map(|&value| finite_f64(value))
                .collect::<Result<Vec<_>, _>>()?,
            TensorData::I32(values) => values.iter().map(|&value| f64::from(value)).collect(),
            TensorData::I64(values) => values
                .iter()
                .map(|&value| {
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "the f64 oracle intentionally models nominal i64-to-f64 conversion"
                    )]
                    let lifted = value as f64;
                    lifted
                })
                .collect(),
            TensorData::Bool(_) => {
                return Err(EvalError::UnsupportedDtype { dtype: DType::Bool });
            }
        };

        Self::new(shape.clone(), lifted)
    }

    /// Creates an `f64` tensor from test or caller-provided data.
    ///
    /// Unlike constants in graph IR, inputs may include NaN or infinity. The
    /// payload is only checked against the supplied shape.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DataShapeMismatch`] if the payload length does not
    /// match the shape element count.
    pub fn from_f64(shape: Shape, values: &[f64]) -> Result<Self, EvalError> {
        Self::new(shape, values.to_vec())
    }

    /// Creates an `f64` tensor by converting integer values.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DataShapeMismatch`] if the payload length does not
    /// match the shape element count.
    pub fn from_i64(shape: Shape, values: &[i64]) -> Result<Self, EvalError> {
        let data = values
            .iter()
            .map(|&value| {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "the f64 oracle intentionally models nominal i64-to-f64 conversion"
                )]
                let lifted = value as f64;
                lifted
            })
            .collect();
        Self::new(shape, data)
    }

    /// Compares two tensors with oracle bit semantics.
    ///
    /// Shapes must match. Finite values and infinities compare by `to_bits`.
    /// Any NaN compares equal to any other NaN, ignoring payload differences.
    /// Signed zero remains significant: `+0.0` and `-0.0` are not bit-equal,
    /// so a bit-exact rewrite must preserve that distinction.
    #[must_use]
    pub fn bit_eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.data.len() == other.data.len()
            && self.data.iter().zip(&other.data).all(|(&left, &right)| {
                (left.is_nan() && right.is_nan()) || left.to_bits() == right.to_bits()
            })
    }

    /// Returns this tensor's logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns this tensor's row-major `f64` payload.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns this tensor's logical element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn into_parts(self) -> (Shape, Vec<f64>) {
        (self.shape, self.data)
    }
}

fn finite_f64(value: f64) -> Result<f64, EvalError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EvalError::NonFiniteFloat)
    }
}
