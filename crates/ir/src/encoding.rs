//! Versioned source-graph encoding shared with the Lean graph model.

use std::collections::HashSet;

use serde_json::{Value, json};
use thiserror::Error;

use crate::{DType, Graph, NodeId, Op, Shape, TensorData, TypeError, infer_types};

/// Errors produced when a graph is outside the version-one source format.
#[derive(Debug, Error)]
pub enum GraphEncodeError {
    /// Structural validation or type inference rejected the source graph.
    #[error(transparent)]
    InvalidGraph(#[from] TypeError),
    /// A source node has a dtype other than `f64`.
    #[error("graph V1 requires f64, but node {node} has dtype {dtype}")]
    UnsupportedDtype {
        /// Rejected node.
        node: NodeId,
        /// Rejected element type.
        dtype: DType,
    },
    /// A source operation is not part of the initial graph format.
    #[error("graph V1 does not support {op} at node {node}")]
    UnsupportedOp {
        /// Rejected node.
        node: NodeId,
        /// Stable operation name.
        op: &'static str,
    },
    /// A tensor dimension is zero.
    #[error("graph V1 requires positive dimensions: node {node}, axis {axis}")]
    ZeroDimension {
        /// Rejected node.
        node: NodeId,
        /// Zero-based dimension index.
        axis: usize,
    },
    /// A source size or identifier cannot be represented as an unsigned 64-bit integer.
    #[error("graph V1 source integer exceeds the unsigned 64-bit range")]
    IntegerOutOfRange,
    /// A constant contains NaN or infinity.
    #[error("graph V1 requires finite constants: node {node}, element {index}")]
    NonFiniteConstant {
        /// Rejected constant node.
        node: NodeId,
        /// Zero-based flat element index.
        index: usize,
    },
    /// Two input nodes have the same name.
    #[error("graph V1 input name occurs more than once: {name:?}")]
    DuplicateInputName {
        /// Repeated input name.
        name: String,
    },
    /// The graph input list differs from input-node appearance order.
    #[error("graph V1 input list must match input-node appearance order")]
    InputOrderMismatch,
    /// JSON serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Encodes a validated source graph as deterministic, compact UTF-8 JSON.
///
/// The positional wire format is `["vtc.graph",1,[nodes],[outputs]]`. Nodes use
/// zero-based arena ids and are emitted in arena order; inputs appear in that
/// same order. Supported nodes are `input`, `const`, `add`, and `relu`, all
/// with `f64` element type. Constant values are exactly 16 lowercase hexadecimal
/// digits containing their IEEE bits, without a `0x` prefix. Output order and
/// duplicate output references are preserved. Names are ordinary JSON strings.
///
/// Dimensions must be positive, while rank-zero scalar shapes are allowed.
/// Shape products must fit both Rust `usize` and the V1 unsigned 64-bit size
/// domain. This source-format check does not establish later signed-index or
/// allocation limits, numerical correctness, or a proof certificate.
///
/// # Errors
///
/// Returns [`GraphEncodeError`] for malformed or ill-typed graphs, unsupported
/// nodes/dtypes, zero dimensions, overflowing shapes, non-finite constants,
/// duplicate input names, an inconsistent input list, or serialization failure.
pub fn encode_graph_v1(graph: &Graph) -> Result<String, GraphEncodeError> {
    let types = infer_types(graph)?;
    let mut nodes = Vec::with_capacity(graph.num_nodes());
    let mut inputs = Vec::new();
    let mut input_names = HashSet::new();

    for index in 0..graph.num_nodes() {
        let id = NodeId::new(index);
        let op = graph.node(id).map_err(TypeError::from)?.op();
        // Inference produces one type for every validated arena node.
        let ty = types.type_of(id).ok_or_else(|| {
            TypeError::from(crate::IrError::InvalidNodeId {
                id,
                num_nodes: graph.num_nodes(),
            })
        })?;
        if ty.dtype() != DType::F64 {
            return Err(GraphEncodeError::UnsupportedDtype {
                node: id,
                dtype: ty.dtype(),
            });
        }
        let dims = encode_shape(id, ty.shape())?;
        let encoded = match op {
            Op::Input { name, .. } => {
                if !input_names.insert(name.as_str()) {
                    return Err(GraphEncodeError::DuplicateInputName { name: name.clone() });
                }
                inputs.push(id);
                json!(["input", "f64", name, dims])
            }
            Op::Const {
                data: TensorData::F64(values),
                ..
            } => {
                let bits = values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        if value.is_finite() {
                            Ok(format!("{:016x}", value.to_bits()))
                        } else {
                            Err(GraphEncodeError::NonFiniteConstant { node: id, index })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                json!(["const", "f64", dims, bits])
            }
            Op::Add(left, right) => json!(["add", wire_id(*left)?, wire_id(*right)?]),
            Op::Relu(input) => json!(["relu", wire_id(*input)?]),
            _ => {
                return Err(GraphEncodeError::UnsupportedOp {
                    node: id,
                    op: op.op_name(),
                });
            }
        };
        nodes.push(encoded);
    }

    if inputs != graph.inputs() {
        return Err(GraphEncodeError::InputOrderMismatch);
    }
    let outputs = graph
        .outputs()
        .iter()
        .copied()
        .map(wire_id)
        .collect::<Result<Vec<_>, _>>()?;
    let encoded: Value = json!(["vtc.graph", 1, nodes, outputs]);
    Ok(serde_json::to_string(&encoded)?)
}

fn encode_shape(node: NodeId, shape: &Shape) -> Result<Vec<u64>, GraphEncodeError> {
    for (axis, dim) in shape.dims().iter().enumerate() {
        if dim.get() == 0 {
            return Err(GraphEncodeError::ZeroDimension { node, axis });
        }
    }
    let numel = shape.numel().map_err(TypeError::from)?;
    u64::try_from(numel).map_err(|_| GraphEncodeError::IntegerOutOfRange)?;
    shape
        .dims()
        .iter()
        .map(|dim| u64::try_from(dim.get()).map_err(|_| GraphEncodeError::IntegerOutOfRange))
        .collect()
}

fn wire_id(id: NodeId) -> Result<u64, GraphEncodeError> {
    u64::try_from(id.index()).map_err(|_| GraphEncodeError::IntegerOutOfRange)
}

#[cfg(test)]
mod tests {
    use crate::{DType, Graph, Node, NodeId, Op, Shape, TensorType};

    use super::{GraphEncodeError, encode_graph_v1};

    #[test]
    fn rejects_inconsistent_input_lists() {
        let input = Node::new(Op::Input {
            name: "x".to_owned(),
            ty: TensorType::new(DType::F64, Shape::new(vec![])),
        });
        for inputs in [vec![], vec![NodeId::new(0), NodeId::new(0)]] {
            let graph = Graph::new(vec![input.clone()], inputs, vec![NodeId::new(0)]);
            assert!(matches!(
                encode_graph_v1(&graph),
                Err(GraphEncodeError::InputOrderMismatch)
            ));
        }
    }

    #[test]
    fn rejects_malformed_graphs_before_serialization() {
        let graph = Graph::new(
            vec![Node::new(Op::Relu(NodeId::new(0)))],
            vec![],
            vec![NodeId::new(0)],
        );
        assert!(matches!(
            encode_graph_v1(&graph),
            Err(GraphEncodeError::InvalidGraph(_))
        ));
    }
}
