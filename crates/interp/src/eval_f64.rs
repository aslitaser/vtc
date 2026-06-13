//! IEEE `f64` graph evaluation with deterministic accumulation order.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use vtc_ir::{DType, Graph, NodeId, Op, Shape, TensorType, TypeError, infer_types};

use crate::TensorF64;
use crate::eval::EvalError;
use crate::tensor::{flat_to_multi, multi_to_flat, shape_numel};

/// Evaluates a graph over IEEE `f64` values.
///
/// This is not the true mathematical semantics. It is a hardware-faithful
/// oracle for bit-exactness tests. Matmul accumulates with `k` ascending for
/// each output element. Sum visits input flat indices in ascending row-major
/// order and scatter-adds into the output.
///
/// # Errors
///
/// Returns [`EvalError`] if the graph is ill-typed, required inputs are absent,
/// input tensors do not match declarations, unsupported dtypes appear, a
/// constant is non-finite, or an internal consistency check fails.
pub fn eval_f64<S: BuildHasher>(
    graph: &Graph,
    inputs: &HashMap<String, TensorF64, S>,
) -> Result<Vec<TensorF64>, EvalError> {
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
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
    values: &[Option<TensorF64>],
) -> Result<TensorF64, EvalError> {
    match op {
        Op::Input { name, ty } => eval_input(name, ty, inputs),
        Op::Const { data, shape } => TensorF64::from_data(data, shape),
        Op::Add(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |left, right| left + right,
        ),
        Op::Sub(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |left, right| left - right,
        ),
        Op::Mul(left, right) => elementwise_binary(
            tensor_at(values, *left)?,
            tensor_at(values, *right)?,
            |left, right| left * right,
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
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
) -> Result<TensorF64, EvalError> {
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
    left: &TensorF64,
    right: &TensorF64,
    mut op: impl FnMut(f64, f64) -> f64,
) -> Result<TensorF64, EvalError> {
    if left.shape() != right.shape() || left.numel() != right.numel() {
        return Err(EvalError::Internal(
            "elementwise operands have inconsistent shapes".to_owned(),
        ));
    }

    let data = left
        .data()
        .iter()
        .zip(right.data())
        .map(|(&left, &right)| op(left, right))
        .collect();
    TensorF64::new(left.shape().clone(), data)
}

fn eval_neg(input: &TensorF64) -> Result<TensorF64, EvalError> {
    let data = input.data().iter().map(|&value| -value).collect();
    TensorF64::new(input.shape().clone(), data)
}

fn eval_relu(input: &TensorF64) -> Result<TensorF64, EvalError> {
    let data = input
        .data()
        .iter()
        .map(|&value| if value > 0.0 { value } else { 0.0 })
        .collect();
    TensorF64::new(input.shape().clone(), data)
}

fn eval_matmul(left: &TensorF64, right: &TensorF64) -> Result<TensorF64, EvalError> {
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

    let mut data = vec![0.0; m.saturating_mul(n)];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                let lhs = f64_at(left, row * k + inner)?;
                let rhs = f64_at(right, inner * n + col)?;
                acc += lhs * rhs;
            }
            let output_index = row * n + col;
            let output = data.get_mut(output_index).ok_or_else(|| {
                EvalError::Internal("matmul output index out of range".to_owned())
            })?;
            *output = acc;
        }
    }

    TensorF64::new(Shape::new(vec![*m_dim, *n_dim]), data)
}

fn eval_sum(input: &TensorF64, axes: &[usize], keepdim: bool) -> Result<TensorF64, EvalError> {
    let output_shape = sum_output_shape(input.shape(), axes, keepdim);
    let mut output_data = vec![0.0; shape_numel(&output_shape)?];
    let rank = input.shape().rank();
    let reduced_axes: HashSet<_> = axes.iter().copied().collect();

    for (flat, value) in input.data().iter().copied().enumerate() {
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

    TensorF64::new(output_shape, output_data)
}

fn eval_reshape(input: &TensorF64, new_shape: &Shape) -> Result<TensorF64, EvalError> {
    let (_, data) = input.clone().into_parts();
    TensorF64::new(new_shape.clone(), data)
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

fn f64_at(tensor: &TensorF64, index: usize) -> Result<f64, EvalError> {
    tensor
        .data()
        .get(index)
        .copied()
        .ok_or_else(|| EvalError::Internal("tensor index out of range".to_owned()))
}

fn tensor_at(values: &[Option<TensorF64>], id: NodeId) -> Result<&TensorF64, EvalError> {
    values
        .get(id.index())
        .and_then(Option::as_ref)
        .ok_or_else(|| EvalError::Internal(format!("missing computed value for node {id}")))
}

fn set_value(
    values: &mut [Option<TensorF64>],
    id: NodeId,
    value: TensorF64,
) -> Result<(), EvalError> {
    let slot = values
        .get_mut(id.index())
        .ok_or_else(|| EvalError::Internal(format!("computed node {id} is out of range")))?;
    *slot = Some(value);
    Ok(())
}
