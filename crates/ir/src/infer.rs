//! Pure graph type inference and per-op shape validation.

use std::collections::HashSet;

use thiserror::Error;

use crate::{DType, Dim, Graph, IrError, NodeId, Op, Shape, TensorType};

/// Out-of-line inferred tensor types for graph nodes.
///
/// `types[i]` is the inferred output type of `NodeId(i)`.
#[derive(Debug, Clone)]
pub struct GraphTypes {
    types: Vec<TensorType>,
}

impl GraphTypes {
    /// Returns the inferred output type for `id`.
    ///
    /// This is deliberately non-panicking so callers can handle side tables that
    /// do not correspond to a particular graph.
    #[must_use]
    pub fn type_of(&self, id: NodeId) -> Option<&TensorType> {
        self.types.get(id.index())
    }

    /// Returns inferred output types for the graph's marked outputs.
    ///
    /// # Errors
    ///
    /// Returns a structural error if an output id does not have a corresponding
    /// inferred type in this side table.
    pub fn output_types<'a>(&'a self, graph: &Graph) -> Result<Vec<&'a TensorType>, TypeError> {
        graph
            .outputs()
            .iter()
            .map(|&output| {
                self.type_of(output)
                    .ok_or(TypeError::Structural(IrError::InvalidNodeId {
                        id: output,
                        num_nodes: self.types.len(),
                    }))
            })
            .collect()
    }
}

/// Errors produced by graph type inference.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TypeError {
    /// Two operands to an op have different shapes.
    #[error("{op} shape mismatch: lhs {lhs}, rhs {rhs}")]
    ShapeMismatch {
        /// Operation being inferred.
        op: &'static str,
        /// Left-hand operand shape.
        lhs: Shape,
        /// Right-hand operand shape.
        rhs: Shape,
    },

    /// Two operands to an op have different dtypes.
    #[error("{op} dtype mismatch: lhs {lhs}, rhs {rhs}")]
    DTypeMismatch {
        /// Operation being inferred.
        op: &'static str,
        /// Left-hand operand dtype.
        lhs: DType,
        /// Right-hand operand dtype.
        rhs: DType,
    },

    /// An op that requires numeric dtype received `bool`.
    #[error("{op} requires numeric dtype, got {dtype}")]
    NonNumericDType {
        /// Operation being inferred.
        op: &'static str,
        /// Non-numeric dtype.
        dtype: DType,
    },

    /// Matmul operands are not both rank-2 tensors.
    #[error("matmul requires rank-2 operands, got lhs rank {lhs_rank}, rhs rank {rhs_rank}")]
    MatmulRank {
        /// Left-hand operand rank.
        lhs_rank: usize,
        /// Right-hand operand rank.
        rhs_rank: usize,
    },

    /// Matmul contraction dimensions differ.
    #[error("matmul contraction mismatch: lhs {lhs}, rhs {rhs}")]
    MatmulContraction {
        /// Left-hand operand shape.
        lhs: Shape,
        /// Right-hand operand shape.
        rhs: Shape,
    },

    /// A sum axis is outside the input rank.
    #[error("sum axis {axis} is out of range for rank {rank}")]
    SumAxisOutOfRange {
        /// Invalid axis.
        axis: usize,
        /// Input rank.
        rank: usize,
    },

    /// A sum axis appears more than once.
    #[error("sum axis {axis} appears more than once")]
    SumDuplicateAxis {
        /// Duplicate axis.
        axis: usize,
    },

    /// Reshape source and target element counts differ.
    #[error("reshape changes element count from {from} to {to}")]
    ReshapeNumelMismatch {
        /// Source element count.
        from: usize,
        /// Target element count.
        to: usize,
    },

    /// Structural graph validation failed before type inference.
    #[error(transparent)]
    Structural(#[from] IrError),
}

/// Infers every graph node's output tensor type.
///
/// This pass is pure: it validates the graph and returns an out-of-line
/// [`GraphTypes`] side table without mutating the graph.
///
/// # Errors
///
/// Returns [`TypeError`] for structural graph errors or per-op type and shape
/// validation failures.
pub fn infer_types(graph: &Graph) -> Result<GraphTypes, TypeError> {
    graph.validate_structure()?;

    let mut types = Vec::with_capacity(graph.num_nodes());
    for index in 0..graph.num_nodes() {
        let id = NodeId::new(index);
        let node = graph.node(id)?;
        let ty = infer_node_type(node.op(), &types)?;
        types.push(ty);
    }

    Ok(GraphTypes { types })
}

fn infer_node_type(op: &Op, types: &[TensorType]) -> Result<TensorType, TypeError> {
    match op {
        // Input nodes trust the declared tensor type.
        Op::Input { ty, .. } => Ok(ty.clone()),

        // Const nodes use payload dtype and declared logical shape.
        Op::Const { data, shape } => Ok(TensorType::new(data.dtype(), shape.clone())),

        // Elementwise binary ops require exactly equal shape and dtype. No
        // broadcasting or dtype promotion is supported.
        Op::Add(left, right) | Op::Sub(left, right) | Op::Mul(left, right) => {
            infer_elementwise_binary(op.op_name(), *left, *right, types)
        }

        // Elementwise unary ops preserve type and reject non-numeric dtypes.
        Op::Neg(input) | Op::Relu(input) => infer_numeric_unary(op.op_name(), *input, types),

        // Matmul is rank-2 only: (M, K) x (K, N) -> (M, N), with matching
        // numeric dtype and no implicit casts.
        Op::Matmul(left, right) => infer_matmul(*left, *right, types),

        // Sum reduces explicit axes. Empty axes are identity so step 4's
        // evaluator should return the input value unchanged for that case.
        Op::Sum {
            input,
            axes,
            keepdim,
        } => infer_sum(*input, axes, *keepdim, types),

        // Reshape preserves dtype and requires exact element-count equality.
        Op::Reshape { input, new_shape } => infer_reshape(*input, new_shape, types),
    }
}

fn infer_elementwise_binary(
    op: &'static str,
    left: NodeId,
    right: NodeId,
    types: &[TensorType],
) -> Result<TensorType, TypeError> {
    let left_ty = operand_type(types, left)?;
    let right_ty = operand_type(types, right)?;

    if left_ty.shape() != right_ty.shape() {
        return Err(TypeError::ShapeMismatch {
            op,
            lhs: left_ty.shape().clone(),
            rhs: right_ty.shape().clone(),
        });
    }
    if left_ty.dtype() != right_ty.dtype() {
        return Err(TypeError::DTypeMismatch {
            op,
            lhs: left_ty.dtype(),
            rhs: right_ty.dtype(),
        });
    }
    ensure_numeric(op, left_ty.dtype())?;

    Ok(left_ty.clone())
}

fn infer_numeric_unary(
    op: &'static str,
    input: NodeId,
    types: &[TensorType],
) -> Result<TensorType, TypeError> {
    let input_ty = operand_type(types, input)?;
    ensure_numeric(op, input_ty.dtype())?;
    Ok(input_ty.clone())
}

fn infer_matmul(
    left: NodeId,
    right: NodeId,
    types: &[TensorType],
) -> Result<TensorType, TypeError> {
    let left_ty = operand_type(types, left)?;
    let right_ty = operand_type(types, right)?;

    let lhs_rank = left_ty.shape().rank();
    let rhs_rank = right_ty.shape().rank();
    if lhs_rank != 2 || rhs_rank != 2 {
        return Err(TypeError::MatmulRank { lhs_rank, rhs_rank });
    }
    if left_ty.dtype() != right_ty.dtype() {
        return Err(TypeError::DTypeMismatch {
            op: "matmul",
            lhs: left_ty.dtype(),
            rhs: right_ty.dtype(),
        });
    }
    ensure_numeric("matmul", left_ty.dtype())?;

    let lhs_dims = left_ty.shape().dims();
    let rhs_dims = right_ty.shape().dims();
    if lhs_dims[1] != rhs_dims[0] {
        return Err(TypeError::MatmulContraction {
            lhs: left_ty.shape().clone(),
            rhs: right_ty.shape().clone(),
        });
    }

    Ok(TensorType::new(
        left_ty.dtype(),
        Shape::new(vec![lhs_dims[0], rhs_dims[1]]),
    ))
}

fn infer_sum(
    input: NodeId,
    axes: &[usize],
    keepdim: bool,
    types: &[TensorType],
) -> Result<TensorType, TypeError> {
    let input_ty = operand_type(types, input)?;
    ensure_numeric("sum", input_ty.dtype())?;

    let rank = input_ty.shape().rank();
    let mut reduced = vec![false; rank];
    let mut seen = HashSet::with_capacity(axes.len());
    for &axis in axes {
        if axis >= rank {
            return Err(TypeError::SumAxisOutOfRange { axis, rank });
        }
        if !seen.insert(axis) {
            return Err(TypeError::SumDuplicateAxis { axis });
        }
        reduced[axis] = true;
    }

    let dims = input_ty.shape().dims();
    let output_dims = if keepdim {
        dims.iter()
            .copied()
            .enumerate()
            .map(
                |(index, dim)| {
                    if reduced[index] { Dim::new(1) } else { dim }
                },
            )
            .collect()
    } else {
        dims.iter()
            .copied()
            .enumerate()
            .filter_map(|(index, dim)| (!reduced[index]).then_some(dim))
            .collect()
    };

    Ok(TensorType::new(input_ty.dtype(), Shape::new(output_dims)))
}

fn infer_reshape(
    input: NodeId,
    new_shape: &Shape,
    types: &[TensorType],
) -> Result<TensorType, TypeError> {
    let input_ty = operand_type(types, input)?;
    let from = input_ty.shape().numel()?;
    let to = new_shape.numel()?;
    if from != to {
        return Err(TypeError::ReshapeNumelMismatch { from, to });
    }

    Ok(TensorType::new(input_ty.dtype(), new_shape.clone()))
}

fn ensure_numeric(op: &'static str, dtype: DType) -> Result<(), TypeError> {
    if dtype.is_numeric() {
        Ok(())
    } else {
        Err(TypeError::NonNumericDType { op, dtype })
    }
}

fn operand_type(types: &[TensorType], id: NodeId) -> Result<&TensorType, TypeError> {
    types
        .get(id.index())
        .ok_or(TypeError::Structural(IrError::InvalidNodeId {
            id,
            num_nodes: types.len(),
        }))
}
