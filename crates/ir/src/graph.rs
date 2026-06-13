//! Arena-backed graph storage and structural validation.

use std::collections::VecDeque;
use std::fmt;

use crate::{IrError, Op};

/// Stable node id into a graph's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct NodeId(usize);

impl NodeId {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the node's arena index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A graph node.
///
/// Types are computed by `infer_types` and stored out-of-line in `GraphTypes`.
/// The IR is never mutated by analysis; later rewrites follow the same pattern
/// by building new graphs rather than changing existing nodes in place.
#[derive(Debug, Clone)]
pub struct Node {
    op: Op,
}

impl Node {
    pub(crate) const fn new(op: Op) -> Self {
        Self { op }
    }

    /// Returns the operation stored in this node.
    #[must_use]
    pub const fn op(&self) -> &Op {
        &self.op
    }
}

/// Arena-backed graph DAG with explicit inputs and outputs.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub(crate) const fn new(nodes: Vec<Node>, inputs: Vec<NodeId>, outputs: Vec<NodeId>) -> Self {
        Self {
            nodes,
            inputs,
            outputs,
        }
    }

    /// Returns a node by id.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] if `id` is outside the node arena.
    pub fn node(&self, id: NodeId) -> Result<&Node, IrError> {
        self.nodes.get(id.index()).ok_or(IrError::InvalidNodeId {
            id,
            num_nodes: self.nodes.len(),
        })
    }

    /// Returns the number of nodes in the arena.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns graph input node ids.
    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Returns graph output node ids.
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Validates graph structure without checking shape compatibility.
    ///
    /// This checks node id ranges, build-order references, input list contents,
    /// output list contents, and constant payload lengths. It deliberately does
    /// not check elementwise shape equality, matmul dimensions, sum axes, or
    /// reshape element-count equality.
    ///
    /// # Errors
    ///
    /// Returns the first structural violation as an [`IrError`].
    pub fn validate_structure(&self) -> Result<(), IrError> {
        for (node_index, node) in self.nodes.iter().enumerate() {
            let node_id = NodeId::new(node_index);
            for operand in node.op.operands() {
                if operand.index() >= self.nodes.len() {
                    return Err(IrError::InvalidNodeId {
                        id: operand,
                        num_nodes: self.nodes.len(),
                    });
                }
                if operand.index() >= node_index {
                    return Err(IrError::ForwardReference {
                        node: node_id,
                        operand,
                    });
                }
            }
        }

        for &input in &self.inputs {
            match self.node(input)?.op() {
                Op::Input { .. } => {}
                _ => return Err(IrError::NotAnInput(input)),
            }
        }

        if self.outputs.is_empty() {
            return Err(IrError::NoOutputs);
        }

        for &output in &self.outputs {
            self.node(output)?;
        }

        for node in &self.nodes {
            if let Op::Const { data, shape } = node.op() {
                let expected = shape.numel()?;
                let got = data.len();
                if got != expected {
                    return Err(IrError::ConstDataShapeMismatch { expected, got });
                }
            }
        }

        Ok(())
    }

    /// Returns a topological ordering of every node in the graph.
    ///
    /// This uses Kahn's algorithm over operand edges. Unlike
    /// [`Self::validate_structure`], it accepts out-of-build-order DAGs so tests
    /// and future importers can validate arbitrary arena orderings.
    ///
    /// # Errors
    ///
    /// Returns [`IrError::InvalidNodeId`] for dangling operands or
    /// [`IrError::Cycle`] if the graph is not a DAG.
    pub fn topo_order(&self) -> Result<Vec<NodeId>, IrError> {
        let num_nodes = self.nodes.len();
        let mut successors = vec![Vec::new(); num_nodes];
        let mut indegrees = vec![0usize; num_nodes];

        for (node_index, node) in self.nodes.iter().enumerate() {
            for operand in node.op.operands() {
                let operand_index = operand.index();
                if operand_index >= num_nodes {
                    return Err(IrError::InvalidNodeId {
                        id: operand,
                        num_nodes,
                    });
                }
                successors[operand_index].push(node_index);
                indegrees[node_index] += 1;
            }
        }

        let mut ready = VecDeque::new();
        for (node_index, indegree) in indegrees.iter().copied().enumerate() {
            if indegree == 0 {
                ready.push_back(node_index);
            }
        }

        let mut order = Vec::with_capacity(num_nodes);
        while let Some(node_index) = ready.pop_front() {
            order.push(NodeId::new(node_index));
            for &successor in &successors[node_index] {
                indegrees[successor] -= 1;
                if indegrees[successor] == 0 {
                    ready.push_back(successor);
                }
            }
        }

        if order.len() == num_nodes {
            Ok(order)
        } else {
            Err(IrError::Cycle)
        }
    }

    /// Renders the graph in a stable SSA-like text format.
    #[must_use]
    pub fn to_text(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph {{")?;
        for (index, node) in self.nodes.iter().enumerate() {
            write!(f, "  %{index} = ")?;
            format_op(f, node.op())?;
            writeln!(f)?;
        }
        f.write_str("  outputs: ")?;
        for (index, output) in self.outputs.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{output}")?;
        }
        writeln!(f)?;
        f.write_str("}")
    }
}

fn format_op(f: &mut fmt::Formatter<'_>, op: &Op) -> fmt::Result {
    match op {
        Op::Input { name, ty } => write!(f, "input {name:?} : {}{}", ty.dtype(), ty.shape()),
        Op::Const { data, shape } => write!(f, "const {}{}", data.dtype(), shape),
        Op::Add(left, right) => write!(f, "add({left}, {right})"),
        Op::Sub(left, right) => write!(f, "sub({left}, {right})"),
        Op::Mul(left, right) => write!(f, "mul({left}, {right})"),
        Op::Neg(input) => write!(f, "neg({input})"),
        Op::Relu(input) => write!(f, "relu({input})"),
        Op::Matmul(left, right) => write!(f, "matmul({left}, {right})"),
        Op::Sum {
            input,
            axes,
            keepdim,
        } => write!(f, "sum({input}, axes={axes:?}, keepdim={keepdim})"),
        Op::Reshape { input, new_shape } => write!(f, "reshape({input}, {new_shape})"),
    }
}

#[cfg(test)]
mod tests {
    use super::{Graph, Node, NodeId};
    use crate::{DType, Dim, IrError, Op, Shape, TensorData, TensorType};

    fn input(name: &str) -> Node {
        Node::new(Op::Input {
            name: name.to_owned(),
            ty: TensorType::new(DType::F32, Shape::new(vec![Dim::new(1)])),
        })
    }

    #[test]
    fn validate_structure_rejects_forward_operand() {
        let graph = Graph::new(
            vec![
                input("a"),
                Node::new(Op::Add(NodeId::new(0), NodeId::new(2))),
                input("b"),
            ],
            vec![NodeId::new(0), NodeId::new(2)],
            vec![NodeId::new(1)],
        );

        assert_eq!(
            graph.validate_structure(),
            Err(IrError::ForwardReference {
                node: NodeId::new(1),
                operand: NodeId::new(2),
            })
        );
    }

    #[test]
    fn validate_structure_rejects_dangling_operand() {
        let graph = Graph::new(
            vec![
                input("a"),
                Node::new(Op::Add(NodeId::new(0), NodeId::new(9))),
            ],
            vec![NodeId::new(0)],
            vec![NodeId::new(1)],
        );

        assert_eq!(
            graph.validate_structure(),
            Err(IrError::InvalidNodeId {
                id: NodeId::new(9),
                num_nodes: 2,
            })
        );
    }

    #[test]
    fn numel_reports_overflow() {
        let shape = Shape::new(vec![Dim::new(usize::MAX), Dim::new(2)]);

        assert_eq!(shape.numel(), Err(IrError::ShapeOverflow));
    }

    #[test]
    fn topo_order_accepts_out_of_build_order_dag() {
        let graph = Graph::new(
            vec![
                Node::new(Op::Mul(NodeId::new(1), NodeId::new(2))),
                input("a"),
                input("b"),
            ],
            vec![NodeId::new(1), NodeId::new(2)],
            vec![NodeId::new(0)],
        );

        let order = graph.topo_order();

        assert_eq!(
            order,
            Ok(vec![NodeId::new(1), NodeId::new(2), NodeId::new(0)])
        );
    }

    #[test]
    fn topo_order_reports_cycle() {
        let graph = Graph::new(
            vec![
                Node::new(Op::Neg(NodeId::new(1))),
                Node::new(Op::Neg(NodeId::new(0))),
            ],
            Vec::new(),
            vec![NodeId::new(0)],
        );

        assert_eq!(graph.topo_order(), Err(IrError::Cycle));
    }

    #[test]
    fn validate_structure_rejects_bad_const_length() {
        let graph = Graph::new(
            vec![Node::new(Op::Const {
                data: TensorData::I32(vec![1, 2]),
                shape: Shape::new(vec![Dim::new(3)]),
            })],
            Vec::new(),
            vec![NodeId::new(0)],
        );

        assert_eq!(
            graph.validate_structure(),
            Err(IrError::ConstDataShapeMismatch {
                expected: 3,
                got: 2,
            })
        );
    }
}
