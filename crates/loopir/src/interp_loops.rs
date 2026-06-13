//! Exact-rational interpreter for affine loop kernels.

use std::collections::HashMap;
use std::hash::BuildHasher;

use num_rational::BigRational;
use num_traits::Zero;
use thiserror::Error;
use vtc_interp::{EvalError, Tensor};
use vtc_ir::{DType, Shape};

use crate::{Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopVar, ScalarExpr, Stmt};

/// Errors produced by loop-kernel evaluation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LoopError {
    /// A required named input was not supplied.
    #[error("missing input {0:?}")]
    MissingInput(String),

    /// A supplied input shape differs from the declared kernel input shape.
    #[error("input {name:?} shape mismatch: expected {expected}, got {got}")]
    InputShapeMismatch {
        /// Input name.
        name: String,
        /// Declared shape.
        expected: Shape,
        /// Supplied tensor shape.
        got: Shape,
    },

    /// A buffer access was outside the flat buffer.
    #[error("buffer {buffer} index {index} is out of bounds for {numel} elements")]
    IndexOutOfBounds {
        /// Buffer being accessed.
        buffer: BufferId,
        /// Evaluated flat index.
        index: i64,
        /// Buffer element count.
        numel: usize,
    },

    /// The interpreter does not define semantics for this dtype.
    #[error("unsupported dtype {dtype}")]
    UnsupportedDtype {
        /// Unsupported dtype.
        dtype: DType,
    },

    /// A floating-point literal was NaN or infinity.
    #[error("non-finite floating-point value")]
    NonFiniteFloat,

    /// Internal consistency check failed.
    #[error("internal loop interpreter error: {0}")]
    Internal(String),
}

/// Evaluates an affine loop kernel over exact rationals.
///
/// Inputs and constants are losslessly lifted through `vtc-interp::Tensor`.
/// Every load and store bounds-checks its evaluated flat affine index.
///
/// # Errors
///
/// Returns [`LoopError`] for missing or mismatched inputs, unsupported dtypes,
/// non-finite constants, out-of-bounds accesses, or internal consistency
/// failures.
pub fn eval_loops<S: BuildHasher>(
    kernel: &Kernel,
    inputs: &HashMap<String, Tensor, S>,
) -> Result<Vec<Tensor>, LoopError> {
    let mut buffers = allocate_buffers(kernel, inputs)?;
    let mut env = HashMap::new();
    exec_stmts(kernel.body(), &mut buffers, &mut env)?;
    if kernel.outputs().len() != kernel.output_shapes().len() {
        return Err(LoopError::Internal(
            "kernel output shape metadata length mismatch".to_owned(),
        ));
    }
    kernel
        .outputs()
        .iter()
        .copied()
        .zip(kernel.output_shapes())
        .map(|(output, shape)| tensor_from_buffer(kernel, &buffers, output, shape))
        .collect()
}

fn allocate_buffers(
    kernel: &Kernel,
    inputs: &HashMap<String, Tensor, impl BuildHasher>,
) -> Result<Vec<Vec<BigRational>>, LoopError> {
    kernel
        .buffers()
        .iter()
        .map(|buffer| allocate_buffer(buffer, inputs))
        .collect()
}

fn allocate_buffer(
    buffer: &Buffer,
    inputs: &HashMap<String, Tensor, impl BuildHasher>,
) -> Result<Vec<BigRational>, LoopError> {
    match buffer.role() {
        BufferRole::Input(name) => {
            let input = inputs
                .get(name)
                .ok_or_else(|| LoopError::MissingInput(name.clone()))?;
            if input.shape() != buffer.shape() {
                return Err(LoopError::InputShapeMismatch {
                    name: name.clone(),
                    expected: buffer.shape().clone(),
                    got: input.shape().clone(),
                });
            }
            Ok(input.data().to_vec())
        }
        BufferRole::Const(data) => Tensor::from_data(data, buffer.shape())
            .map(|tensor| tensor.data().to_vec())
            .map_err(loop_error_from_eval),
        BufferRole::Temp | BufferRole::Output => {
            Ok(vec![BigRational::zero(); shape_numel(buffer.shape())?])
        }
    }
}

fn exec_stmts(
    stmts: &[Stmt],
    buffers: &mut [Vec<BigRational>],
    env: &mut HashMap<LoopVar, i64>,
) -> Result<(), LoopError> {
    for stmt in stmts {
        exec_stmt(stmt, buffers, env)?;
    }
    Ok(())
}

fn exec_stmt(
    stmt: &Stmt,
    buffers: &mut [Vec<BigRational>],
    env: &mut HashMap<LoopVar, i64>,
) -> Result<(), LoopError> {
    match stmt {
        Stmt::For { var, lo, hi, body } => {
            let lo = lo.eval(env);
            let hi = hi.eval(env);
            for value in lo..hi {
                env.insert(*var, value);
                exec_stmts(body, buffers, env)?;
            }
            env.remove(var);
            Ok(())
        }
        Stmt::Assign { target, value } => {
            let value = eval_scalar(value, buffers, env)?;
            store(target, value, buffers, env)
        }
    }
}

fn eval_scalar(
    expr: &ScalarExpr,
    buffers: &[Vec<BigRational>],
    env: &HashMap<LoopVar, i64>,
) -> Result<BigRational, LoopError> {
    match expr {
        ScalarExpr::Load(reference) => load(reference, buffers, env),
        ScalarExpr::ConstScalar(value) => Ok(value.clone()),
        ScalarExpr::Add(left, right) => {
            Ok(eval_scalar(left, buffers, env)? + eval_scalar(right, buffers, env)?)
        }
        ScalarExpr::Sub(left, right) => {
            Ok(eval_scalar(left, buffers, env)? - eval_scalar(right, buffers, env)?)
        }
        ScalarExpr::Mul(left, right) => {
            Ok(eval_scalar(left, buffers, env)? * eval_scalar(right, buffers, env)?)
        }
        ScalarExpr::Neg(input) => Ok(-eval_scalar(input, buffers, env)?),
        ScalarExpr::Relu(input) => {
            let value = eval_scalar(input, buffers, env)?;
            if value > BigRational::zero() {
                Ok(value)
            } else {
                Ok(BigRational::zero())
            }
        }
    }
}

fn load(
    reference: &BufferRef,
    buffers: &[Vec<BigRational>],
    env: &HashMap<LoopVar, i64>,
) -> Result<BigRational, LoopError> {
    let index = checked_index(reference, buffers, env)?;
    let buffer = buffers
        .get(buffer_index(reference.buffer)?)
        .ok_or_else(|| LoopError::Internal(format!("missing buffer {}", reference.buffer)))?;
    let index_i64 = i64::try_from(index)
        .map_err(|_| LoopError::Internal("index conversion overflow".to_owned()))?;
    let value = buffer.get(index).ok_or(LoopError::IndexOutOfBounds {
        buffer: reference.buffer,
        index: index_i64,
        numel: buffer.len(),
    })?;
    Ok(value.clone())
}

fn store(
    reference: &BufferRef,
    value: BigRational,
    buffers: &mut [Vec<BigRational>],
    env: &HashMap<LoopVar, i64>,
) -> Result<(), LoopError> {
    let index = checked_index(reference, buffers, env)?;
    let buffer_index = buffer_index(reference.buffer)?;
    let Some(buffer) = buffers.get_mut(buffer_index) else {
        return Err(LoopError::Internal(format!(
            "missing buffer {}",
            reference.buffer
        )));
    };
    let Some(slot) = buffer.get_mut(index) else {
        return Err(LoopError::IndexOutOfBounds {
            buffer: reference.buffer,
            index: i64::try_from(index)
                .map_err(|_| LoopError::Internal("index conversion overflow".to_owned()))?,
            numel: buffer.len(),
        });
    };
    *slot = value;
    Ok(())
}

fn checked_index(
    reference: &BufferRef,
    buffers: &[Vec<BigRational>],
    env: &HashMap<LoopVar, i64>,
) -> Result<usize, LoopError> {
    let buffer_index = buffer_index(reference.buffer)?;
    let Some(buffer) = buffers.get(buffer_index) else {
        return Err(LoopError::Internal(format!(
            "missing buffer {}",
            reference.buffer
        )));
    };
    let index = reference.index.eval(env);
    let Ok(index_usize) = usize::try_from(index) else {
        return Err(LoopError::IndexOutOfBounds {
            buffer: reference.buffer,
            index,
            numel: buffer.len(),
        });
    };
    if index_usize >= buffer.len() {
        return Err(LoopError::IndexOutOfBounds {
            buffer: reference.buffer,
            index,
            numel: buffer.len(),
        });
    }
    Ok(index_usize)
}

fn tensor_from_buffer(
    kernel: &Kernel,
    buffers: &[Vec<BigRational>],
    id: BufferId,
    shape: &Shape,
) -> Result<Tensor, LoopError> {
    kernel
        .buffer(id)
        .ok_or_else(|| LoopError::Internal(format!("missing output buffer {id}")))?;
    let data = buffers
        .get(buffer_index(id)?)
        .ok_or_else(|| LoopError::Internal(format!("missing output data for {id}")))?
        .clone();
    Tensor::new(shape.clone(), data).map_err(loop_error_from_eval)
}

fn loop_error_from_eval(error: EvalError) -> LoopError {
    match error {
        EvalError::MissingInput(name) => LoopError::MissingInput(name),
        EvalError::InputShapeMismatch {
            name,
            expected,
            got,
        } => LoopError::InputShapeMismatch {
            name,
            expected,
            got,
        },
        EvalError::UnsupportedDtype { dtype } => LoopError::UnsupportedDtype { dtype },
        EvalError::NonFiniteFloat => LoopError::NonFiniteFloat,
        EvalError::DataShapeMismatch { expected, got } => LoopError::Internal(format!(
            "tensor data length mismatch: expected {expected}, got {got}"
        )),
        EvalError::Type(error) => LoopError::Internal(error.to_string()),
        EvalError::Internal(message) => LoopError::Internal(message),
    }
}

fn shape_numel(shape: &vtc_ir::Shape) -> Result<usize, LoopError> {
    shape
        .numel()
        .map_err(|_| LoopError::Internal("shape element count overflow".to_owned()))
}

fn buffer_index(id: BufferId) -> Result<usize, LoopError> {
    usize::try_from(id.id()).map_err(|_| LoopError::Internal("buffer id overflow".to_owned()))
}
