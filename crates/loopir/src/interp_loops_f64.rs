//! IEEE `f64` interpreter for affine loop kernels.

use std::collections::HashMap;
use std::hash::BuildHasher;

use num_traits::ToPrimitive;
use vtc_interp::TensorF64;

use crate::interp_loops::{buffer_index, loop_error_from_eval, shape_numel};
use crate::{
    Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopError, LoopVar, ScalarExpr, Stmt,
};

/// Evaluates an affine loop kernel over IEEE `f64` values.
///
/// The interpreter executes statements in the exact order written in the
/// kernel. `For` loops iterate ascending over their affine `[lo, hi)` bounds,
/// and every `Assign` stores immediately. The resulting accumulation order is
/// therefore completely determined by the loop nest.
///
/// # Errors
///
/// Returns [`LoopError`] for missing or mismatched inputs, unsupported
/// constants, non-finite constants, out-of-bounds accesses, or internal
/// consistency failures.
pub fn eval_loops_f64<S: BuildHasher>(
    kernel: &Kernel,
    inputs: &HashMap<String, TensorF64, S>,
) -> Result<Vec<TensorF64>, LoopError> {
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
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
) -> Result<Vec<Vec<f64>>, LoopError> {
    kernel
        .buffers()
        .iter()
        .map(|buffer| allocate_buffer(buffer, inputs))
        .collect()
}

fn allocate_buffer(
    buffer: &Buffer,
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
) -> Result<Vec<f64>, LoopError> {
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
        BufferRole::Const(data) => TensorF64::from_data(data, buffer.shape())
            .map(|tensor| tensor.data().to_vec())
            .map_err(loop_error_from_eval),
        BufferRole::Temp | BufferRole::Output => Ok(vec![0.0; shape_numel(buffer.shape())?]),
    }
}

fn exec_stmts(
    stmts: &[Stmt],
    buffers: &mut [Vec<f64>],
    env: &mut HashMap<LoopVar, i64>,
) -> Result<(), LoopError> {
    for stmt in stmts {
        exec_stmt(stmt, buffers, env)?;
    }
    Ok(())
}

fn exec_stmt(
    stmt: &Stmt,
    buffers: &mut [Vec<f64>],
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
    buffers: &[Vec<f64>],
    env: &HashMap<LoopVar, i64>,
) -> Result<f64, LoopError> {
    match expr {
        ScalarExpr::Load(reference) => load(reference, buffers, env),
        ScalarExpr::ConstScalar(value) => value.to_f64().ok_or_else(|| {
            LoopError::Internal("rational scalar literal is not representable as f64".to_owned())
        }),
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
            if value > 0.0 { Ok(value) } else { Ok(0.0) }
        }
    }
}

fn load(
    reference: &BufferRef,
    buffers: &[Vec<f64>],
    env: &HashMap<LoopVar, i64>,
) -> Result<f64, LoopError> {
    let index = checked_index(reference, buffers, env)?;
    let buffer = buffers
        .get(buffer_index(reference.buffer)?)
        .ok_or_else(|| LoopError::Internal(format!("missing buffer {}", reference.buffer)))?;
    let index_i64 = i64::try_from(index)
        .map_err(|_| LoopError::Internal("index conversion overflow".to_owned()))?;
    buffer
        .get(index)
        .copied()
        .ok_or(LoopError::IndexOutOfBounds {
            buffer: reference.buffer,
            index: index_i64,
            numel: buffer.len(),
        })
}

fn store(
    reference: &BufferRef,
    value: f64,
    buffers: &mut [Vec<f64>],
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
    buffers: &[Vec<f64>],
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
    buffers: &[Vec<f64>],
    id: BufferId,
    shape: &vtc_ir::Shape,
) -> Result<TensorF64, LoopError> {
    kernel
        .buffer(id)
        .ok_or_else(|| LoopError::Internal(format!("missing output buffer {id}")))?;
    let data = buffers
        .get(buffer_index(id)?)
        .ok_or_else(|| LoopError::Internal(format!("missing output data for {id}")))?
        .clone();
    TensorF64::new(shape.clone(), data).map_err(loop_error_from_eval)
}
