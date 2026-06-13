//! Graph operation kinds.

use crate::{NodeId, Shape, TensorData, TensorType};

/// A graph operation with its operands embedded as [`NodeId`] values.
///
/// The op set is intentionally restricted to operators with an exact value over
/// rationals, because the step-4 reference semantics evaluates over exact
/// rationals. Transcendentals such as exp, log, and sin; division; softmax;
/// broadcasting; batched or transposed matmul; and convolution are deliberately
/// excluded for now.
#[derive(Debug, Clone)]
pub enum Op {
    /// Named graph input with a declared tensor type.
    Input {
        /// Input name.
        name: String,
        /// Declared input tensor type.
        ty: TensorType,
    },
    /// Literal tensor constant.
    Const {
        /// Constant payload.
        data: TensorData,
        /// Constant logical shape.
        shape: Shape,
    },
    /// Elementwise addition.
    Add(NodeId, NodeId),
    /// Elementwise subtraction.
    Sub(NodeId, NodeId),
    /// Elementwise multiplication.
    Mul(NodeId, NodeId),
    /// Elementwise negation.
    Neg(NodeId),
    /// Elementwise rectified linear unit.
    Relu(NodeId),
    /// Two-dimensional matrix multiplication, `(M, K) x (K, N)`.
    Matmul(NodeId, NodeId),
    /// Sum reduction over explicit axes.
    Sum {
        /// Reduction input.
        input: NodeId,
        /// Axes to reduce.
        axes: Vec<usize>,
        /// Whether reduced axes remain as size-one dimensions.
        keepdim: bool,
    },
    /// Reshape to a new logical shape.
    Reshape {
        /// Reshape input.
        input: NodeId,
        /// Target shape.
        new_shape: Shape,
    },
}

impl Op {
    /// Returns this op's operand ids in source order.
    #[must_use]
    pub fn operands(&self) -> Vec<NodeId> {
        match self {
            Self::Input { .. } | Self::Const { .. } => Vec::new(),
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Matmul(left, right) => vec![*left, *right],
            Self::Neg(input)
            | Self::Relu(input)
            | Self::Sum { input, .. }
            | Self::Reshape { input, .. } => vec![*input],
        }
    }

    /// Returns a copy of this op with every operand remapped by `f`.
    #[must_use]
    pub fn map_operands(&self, f: impl Fn(NodeId) -> NodeId) -> Self {
        match self {
            Self::Input { name, ty } => Self::Input {
                name: name.clone(),
                ty: ty.clone(),
            },
            Self::Const { data, shape } => Self::Const {
                data: data.clone(),
                shape: shape.clone(),
            },
            Self::Add(left, right) => Self::Add(f(*left), f(*right)),
            Self::Sub(left, right) => Self::Sub(f(*left), f(*right)),
            Self::Mul(left, right) => Self::Mul(f(*left), f(*right)),
            Self::Neg(input) => Self::Neg(f(*input)),
            Self::Relu(input) => Self::Relu(f(*input)),
            Self::Matmul(left, right) => Self::Matmul(f(*left), f(*right)),
            Self::Sum {
                input,
                axes,
                keepdim,
            } => Self::Sum {
                input: f(*input),
                axes: axes.clone(),
                keepdim: *keepdim,
            },
            Self::Reshape { input, new_shape } => Self::Reshape {
                input: f(*input),
                new_shape: new_shape.clone(),
            },
        }
    }

    /// Returns the stable text name for this operation kind.
    #[must_use]
    pub const fn op_name(&self) -> &'static str {
        match self {
            Self::Input { .. } => "input",
            Self::Const { .. } => "const",
            Self::Add(_, _) => "add",
            Self::Sub(_, _) => "sub",
            Self::Mul(_, _) => "mul",
            Self::Neg(_) => "neg",
            Self::Relu(_) => "relu",
            Self::Matmul(_, _) => "matmul",
            Self::Sum { .. } => "sum",
            Self::Reshape { .. } => "reshape",
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Dim, NodeId, Op, Shape};

    #[test]
    fn map_operands_rewrites_all_node_ids() {
        let op = Op::Sum {
            input: NodeId::new(2),
            axes: vec![0],
            keepdim: true,
        };

        let mapped = op.map_operands(|id| NodeId::new(id.index() + 10));

        assert_eq!(mapped.operands(), vec![NodeId::new(12)]);
    }

    #[test]
    fn map_operands_preserves_non_operand_data() {
        let op = Op::Reshape {
            input: NodeId::new(3),
            new_shape: Shape::new(vec![Dim::new(2), Dim::new(2)]),
        };

        let mapped = op.map_operands(|_| NodeId::new(0));

        match mapped {
            Op::Reshape { input, new_shape } => {
                assert_eq!(input, NodeId::new(0));
                assert_eq!(new_shape, Shape::new(vec![Dim::new(2), Dim::new(2)]));
            }
            _ => unreachable!("map_operands preserves the op variant"),
        }
    }
}
