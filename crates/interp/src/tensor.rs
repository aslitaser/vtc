//! Runtime tensor values for the exact-rational interpreter.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, float::FloatCore};
use vtc_ir::{DType, Shape, TensorData};

use crate::EvalError;

/// Runtime tensor value with row-major exact-rational data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<BigRational>,
}

impl Tensor {
    /// Creates a tensor from a shape and exact-rational payload.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DataShapeMismatch`] if the payload length does not
    /// match the shape element count.
    pub fn new(shape: Shape, data: Vec<BigRational>) -> Result<Self, EvalError> {
        let expected = shape_numel(&shape)?;
        let got = data.len();
        if expected != got {
            return Err(EvalError::DataShapeMismatch { expected, got });
        }

        Ok(Self { shape, data })
    }

    /// Losslessly lifts nominal tensor data into exact rationals.
    ///
    /// Integer values become `value / 1`. Finite floats become their exact
    /// dyadic rational value. Booleans are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] if the payload length mismatches the shape, a
    /// float is not finite, or the dtype is unsupported.
    pub fn from_data(data: &TensorData, shape: &Shape) -> Result<Self, EvalError> {
        let lifted = match data {
            TensorData::F32(values) => values
                .iter()
                .map(|&value| rational_from_f64(f64::from(value)))
                .collect::<Result<Vec<_>, _>>()?,
            TensorData::F64(values) => values
                .iter()
                .map(|&value| rational_from_f64(value))
                .collect::<Result<Vec<_>, _>>()?,
            TensorData::I32(values) => values
                .iter()
                .map(|&value| BigRational::from_integer(BigInt::from(value)))
                .collect(),
            TensorData::I64(values) => values
                .iter()
                .map(|&value| BigRational::from_integer(BigInt::from(value)))
                .collect(),
            TensorData::Bool(_) => {
                return Err(EvalError::UnsupportedDtype { dtype: DType::Bool });
            }
        };

        Self::new(shape.clone(), lifted)
    }

    /// Losslessly lifts `f64` values into an exact-rational tensor.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] if the payload length mismatches the shape or a
    /// float is not finite.
    pub fn from_f64(shape: Shape, values: &[f64]) -> Result<Self, EvalError> {
        let data = values
            .iter()
            .map(|&value| rational_from_f64(value))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(shape, data)
    }

    /// Losslessly lifts `i64` values into an exact-rational tensor.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::DataShapeMismatch`] if the payload length does not
    /// match the shape element count.
    pub fn from_i64(shape: Shape, values: &[i64]) -> Result<Self, EvalError> {
        let data = values
            .iter()
            .map(|&value| BigRational::from_integer(BigInt::from(value)))
            .collect();
        Self::new(shape, data)
    }

    /// Returns this tensor's logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns this tensor's row-major exact-rational payload.
    #[must_use]
    pub fn data(&self) -> &[BigRational] {
        &self.data
    }

    /// Returns this tensor's logical element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn into_parts(self) -> (Shape, Vec<BigRational>) {
        (self.shape, self.data)
    }
}

pub(crate) fn row_major_strides(shape: &Shape) -> Vec<usize> {
    let mut strides = vec![1; shape.rank()];
    let mut stride = 1usize;
    for (index, dim) in shape.dims().iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride.saturating_mul(dim.get());
    }
    strides
}

pub(crate) fn flat_to_multi(flat: usize, shape: &Shape) -> Vec<usize> {
    let strides = row_major_strides(shape);
    shape
        .dims()
        .iter()
        .zip(strides)
        .map(|(dim, stride)| {
            if dim.get() == 0 {
                0
            } else {
                (flat / stride) % dim.get()
            }
        })
        .collect()
}

pub(crate) fn multi_to_flat(indices: &[usize], shape: &Shape) -> usize {
    let strides = row_major_strides(shape);
    indices
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

pub(crate) fn shape_numel(shape: &Shape) -> Result<usize, EvalError> {
    shape
        .numel()
        .map_err(|_| EvalError::Internal("shape element count overflow".to_owned()))
}

fn rational_from_f64(value: f64) -> Result<BigRational, EvalError> {
    if !value.is_finite() {
        return Err(EvalError::NonFiniteFloat);
    }

    let (mantissa, exponent, sign) = value.integer_decode();
    let mut numerator = BigInt::from(mantissa);
    if sign < 0 {
        numerator = -numerator;
    }

    let shift = usize::from(exponent.unsigned_abs());
    if exponent >= 0 {
        Ok(BigRational::from_integer(numerator << shift))
    } else {
        let denominator = BigInt::one() << shift;
        Ok(BigRational::new(numerator, denominator))
    }
}
