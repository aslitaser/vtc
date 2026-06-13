//! Exact-rational graph evaluation.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use num_rational::BigRational;
use num_traits::Zero;
use thiserror::Error;
use vtc_ir::{DType, Graph, NodeId, Op, Shape, TensorType, TypeError, infer_types};

use crate::Tensor;
use crate::tensor::{flat_to_multi, multi_to_flat, shape_numel};

/// Errors produced by exact-rational graph evaluation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EvalError {
    /// Type inference rejected the graph before evaluation.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// A required named input was not supplied.
    #[error("missing input {0:?}")]
    MissingInput(String),

    /// A supplied input shape differs from the declared graph input shape.
    #[error("input {name:?} shape mismatch: expected {expected}, got {got}")]
    InputShapeMismatch {
        /// Input name.
        name: String,
        /// Declared shape.
        expected: Shape,
        /// Supplied tensor shape.
        got: Shape,
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

    /// Runtime tensor data length does not match its shape.
    #[error("tensor data length mismatch: expected {expected}, got {got}")]
    DataShapeMismatch {
        /// Expected element count from the shape.
        expected: usize,
        /// Actual data element count.
        got: usize,
    },

    /// Internal consistency check failed.
    #[error("internal interpreter error: {0}")]
    Internal(String),
}

/// Evaluates a graph over exact rationals.
///
/// The graph is type-checked first. Evaluation is intentionally direct and
/// unoptimized; it is the reference oracle for future differential tests.
///
/// # Errors
///
/// Returns [`EvalError`] if the graph is ill-typed, required inputs are absent,
/// input tensors do not match declarations, unsupported dtypes appear, or an
/// internal consistency check fails.
pub fn eval<S: BuildHasher>(
    graph: &Graph,
    inputs: &HashMap<String, Tensor, S>,
) -> Result<Vec<Tensor>, EvalError> {
    let graph_types = infer_types(graph)?;
    let order = graph.topo_order().map_err(TypeError::from)?;
    let mut values = vec![None; graph.num_nodes()];

    for node_id in order {
        let node = graph.node(node_id).map_err(TypeError::from)?;
        let value = eval_node(node.op(), inputs, &values)?;
        let inferred = graph_types.type_of(node_id).ok_or_else(|| {
            EvalError::Internal(format!("missing inferred type for node {node_id}"))
        })?;
        if value.shape() != inferred.shape() {
            return Err(EvalError::Internal(format!(
                "node {node_id} produced shape {}, expected {}",
                value.shape(),
                inferred.shape()
            )));
        }
        set_value(&mut values, node_id, value)?;
    }

    graph
        .outputs()
        .iter()
        .map(|&output| tensor_at(&values, output).cloned())
        .collect()
}

fn eval_node(
    op: &Op,
    inputs: &HashMap<String, Tensor, impl BuildHasher>,
    values: &[Option<Tensor>],
) -> Result<Tensor, EvalError> {
    match op {
        Op::Input { name, ty } => eval_input(name, ty, inputs),
        Op::Const { data, shape } => Tensor::from_data(data, shape),
        Op::Add(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |a, b| a + b,
        ),
        Op::Sub(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |a, b| a - b,
        ),
        Op::Mul(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |a, b| a * b,
        ),
        Op::Neg(input) => eval_neg(tensor_at(values, *input)?),
        Op::Relu(input) => eval_relu(tensor_at(values, *input)?),
        Op::Matmul(left, right) => {
            eval_matmul(tensor_at(values, *left)?, tensor_at(values, *right)?)
        }
        Op::Sum {
            input,
            axes,
            keepdim,
        } => eval_sum(tensor_at(values, *input)?, axes, *keepdim),
        Op::Reshape { input, new_shape } => eval_reshape(tensor_at(values, *input)?, new_shape),
    }
}

fn eval_input(
    name: &str,
    ty: &TensorType,
    inputs: &HashMap<String, Tensor, impl BuildHasher>,
) -> Result<Tensor, EvalError> {
    let input = inputs
        .get(name)
        .ok_or_else(|| EvalError::MissingInput(name.to_owned()))?;

    if input.shape() != ty.shape() {
        return Err(EvalError::InputShapeMismatch {
            name: name.to_owned(),
            expected: ty.shape().clone(),
            got: input.shape().clone(),
        });
    }
    if ty.dtype() == DType::Bool {
        return Err(EvalError::UnsupportedDtype { dtype: DType::Bool });
    }

    Ok(input.clone())
}

fn elementwise_binary(
    left: &Tensor,
    right: &Tensor,
    mut op: impl FnMut(&BigRational, &BigRational) -> BigRational,
) -> Result<Tensor, EvalError> {
    if left.shape() != right.shape() || left.numel() != right.numel() {
        return Err(EvalError::Internal(
            "elementwise operands have inconsistent shapes".to_owned(),
        ));
    }

    let data = left
        .data()
        .iter()
        .zip(right.data())
        .map(|(left, right)| op(left, right))
        .collect();
    Tensor::new(left.shape().clone(), data)
}

fn eval_neg(input: &Tensor) -> Result<Tensor, EvalError> {
    let data = input.data().iter().map(|value| -value).collect();
    Tensor::new(input.shape().clone(), data)
}

fn eval_relu(input: &Tensor) -> Result<Tensor, EvalError> {
    let zero = BigRational::zero();
    let data = input
        .data()
        .iter()
        .map(|value| {
            if value > &zero {
                value.clone()
            } else {
                zero.clone()
            }
        })
        .collect();
    Tensor::new(input.shape().clone(), data)
}

fn eval_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor, EvalError> {
    let [m_dim, k_dim] = left.shape().dims() else {
        return Err(EvalError::Internal("matmul lhs is not rank 2".to_owned()));
    };
    let [rhs_k_dim, n_dim] = right.shape().dims() else {
        return Err(EvalError::Internal("matmul rhs is not rank 2".to_owned()));
    };

    let m = m_dim.get();
    let k = k_dim.get();
    let rhs_k = rhs_k_dim.get();
    let n = n_dim.get();
    if k != rhs_k {
        return Err(EvalError::Internal(
            "matmul contraction dimensions differ".to_owned(),
        ));
    }

    let mut data = vec![BigRational::zero(); m.saturating_mul(n)];
    for row in 0..m {
        for col in 0..n {
            let mut acc = BigRational::zero();
            for inner in 0..k {
                let lhs = rational_at(left, row * k + inner)?;
                let rhs = rational_at(right, inner * n + col)?;
                acc += lhs * rhs;
            }
            let output_index = row * n + col;
            let output = data.get_mut(output_index).ok_or_else(|| {
                EvalError::Internal("matmul output index out of range".to_owned())
            })?;
            *output = acc;
        }
    }

    Tensor::new(Shape::new(vec![*m_dim, *n_dim]), data)
}

fn eval_sum(input: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor, EvalError> {
    let output_shape = sum_output_shape(input.shape(), axes, keepdim);
    let mut output_data = vec![BigRational::zero(); shape_numel(&output_shape)?];
    let rank = input.shape().rank();
    let reduced_axes: HashSet<_> = axes.iter().copied().collect();

    for (flat, value) in input.data().iter().enumerate() {
        let input_multi = flat_to_multi(flat, input.shape());
        let output_multi: Vec<usize> = if keepdim {
            input_multi
                .iter()
                .copied()
                .enumerate()
                .map(|(axis, index)| {
                    if reduced_axes.contains(&axis) {
                        0
                    } else {
                        index
                    }
                })
                .collect()
        } else {
            input_multi
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(axis, index)| (!reduced_axes.contains(&axis)).then_some(index))
                .collect()
        };

        let output_flat = if rank == axes.len() && !keepdim {
            0
        } else {
            multi_to_flat(&output_multi, &output_shape)
        };
        let output = output_data
            .get_mut(output_flat)
            .ok_or_else(|| EvalError::Internal("sum output index out of range".to_owned()))?;
        *output += value;
    }

    Tensor::new(output_shape, output_data)
}

fn eval_reshape(input: &Tensor, new_shape: &Shape) -> Result<Tensor, EvalError> {
    let (_, data) = input.clone().into_parts();
    Tensor::new(new_shape.clone(), data)
}

fn sum_output_shape(input_shape: &Shape, axes: &[usize], keepdim: bool) -> Shape {
    let reduced_axes: HashSet<_> = axes.iter().copied().collect();
    let dims = if keepdim {
        input_shape
            .dims()
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, dim)| {
                if reduced_axes.contains(&axis) {
                    vtc_ir::Dim::new(1)
                } else {
                    dim
                }
            })
            .collect()
    } else {
        input_shape
            .dims()
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, dim)| (!reduced_axes.contains(&axis)).then_some(dim))
            .collect()
    };
    Shape::new(dims)
}

fn rational_at(tensor: &Tensor, index: usize) -> Result<&BigRational, EvalError> {
    tensor
        .data()
        .get(index)
        .ok_or_else(|| EvalError::Internal("tensor index out of range".to_owned()))
}

fn tensor_at(values: &[Option<Tensor>], id: NodeId) -> Result<&Tensor, EvalError> {
    values
        .get(id.index())
        .and_then(Option::as_ref)
        .ok_or_else(|| EvalError::Internal(format!("missing computed value for node {id}")))
}

fn set_value(values: &mut [Option<Tensor>], id: NodeId, value: Tensor) -> Result<(), EvalError> {
    let slot = values
        .get_mut(id.index())
        .ok_or_else(|| EvalError::Internal(format!("computed node {id} is out of range")))?;
    *slot = Some(value);
    Ok(())
}
