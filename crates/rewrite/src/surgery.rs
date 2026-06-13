//! Functional graph-surgery primitives.

use std::collections::{HashMap, HashSet};

use thiserror::Error;
use vtc_ir::{Graph, GraphBuilder, IrError, NodeId, Op};

/// Errors produced by graph-surgery primitives.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SurgeryError {
    /// A referenced node id is outside the graph.
    #[error("invalid node id {id}")]
    InvalidNodeId {
        /// Invalid node id.
        id: NodeId,
    },

    /// A rewrite would introduce a self-reference or forward reference.
    #[error("node {node} would reference non-ancestor operand {operand}")]
    ForwardReference {
        /// Node being redefined.
        node: NodeId,
        /// Invalid operand.
        operand: NodeId,
    },

    /// The rebuilt graph failed structural validation.
    #[error(transparent)]
    Invalid(#[from] IrError),
}

/// Replaces every use of `target` with `replacement`.
///
/// The target node's own definition is left unchanged. Operands in all nodes
/// and graph outputs are remapped, then the rebuilt graph is structurally
/// validated.
///
/// # Errors
///
/// Returns [`SurgeryError`] if either id is invalid or the rebuilt graph is
/// structurally invalid.
pub fn replace_all_uses(
    graph: &Graph,
    target: NodeId,
    replacement: NodeId,
) -> Result<Graph, SurgeryError> {
    validate_node_id(graph, target)?;
    validate_node_id(graph, replacement)?;

    rebuild_graph(
        graph,
        |old_id, op| {
            if old_id == target {
                op.clone()
            } else {
                op.map_operands(|operand| {
                    if operand == target {
                        replacement
                    } else {
                        operand
                    }
                })
            }
        },
        |output| {
            if output == target {
                replacement
            } else {
                output
            }
        },
    )
}

/// Redefines one node with a new operation.
///
/// The new op may only reference valid operands that appear earlier than the
/// redefined node in index order. The rebuilt graph is structurally validated.
///
/// # Errors
///
/// Returns [`SurgeryError`] if `node` or a new operand is invalid, the new op
/// has a non-ancestor operand, or the rebuilt graph is structurally invalid.
#[allow(
    clippy::needless_pass_by_value,
    reason = "step-6 public API requires redefine_node to take ownership of the replacement Op"
)]
pub fn redefine_node(graph: &Graph, node: NodeId, new_op: Op) -> Result<Graph, SurgeryError> {
    validate_node_id(graph, node)?;
    for operand in new_op.operands() {
        validate_node_id(graph, operand)?;
        if operand.index() >= node.index() {
            return Err(SurgeryError::ForwardReference { node, operand });
        }
    }

    rebuild_graph(
        graph,
        |old_id, op| {
            if old_id == node {
                new_op.clone()
            } else {
                op.clone()
            }
        },
        |output| output,
    )
}

/// Removes dead nodes and rebuilds a compact index-topological graph.
///
/// Reachability starts from graph outputs and follows operands. All declared
/// input nodes are also kept so the input signature remains stable, even when
/// an input is currently unused. Kept nodes are emitted in ascending old index
/// order, which re-establishes the index-topological DAG invariant. Future
/// node-adding rewrites should call this to canonicalize after edits.
///
/// # Errors
///
/// Returns [`SurgeryError`] if the original graph or rebuilt graph is
/// structurally invalid.
pub fn prune_and_rebuild(graph: &Graph) -> Result<Graph, SurgeryError> {
    graph.validate_structure()?;
    let mut keep = HashSet::new();
    for &output in graph.outputs() {
        collect_reachable(graph, output, &mut keep)?;
    }
    keep.extend(graph.inputs().iter().copied());

    let kept_ids = sorted_node_ids(graph)?
        .into_iter()
        .filter(|id| keep.contains(id))
        .collect::<Vec<_>>();

    let mut old_to_new = HashMap::with_capacity(kept_ids.len());
    let mut builder = GraphBuilder::new();
    for old_id in kept_ids {
        let op = graph.node(old_id)?.op();
        let new_id = emit_op(&mut builder, op, &old_to_new)?;
        old_to_new.insert(old_id, new_id);
    }

    for &output in graph.outputs() {
        let mapped = map_old_id(&old_to_new, output)?;
        builder.mark_output(mapped)?;
    }

    Ok(builder.build()?)
}

fn rebuild_graph(
    graph: &Graph,
    mut map_op: impl FnMut(NodeId, &Op) -> Op,
    map_output: impl Fn(NodeId) -> NodeId,
) -> Result<Graph, SurgeryError> {
    graph.validate_structure()?;
    let old_ids = sorted_node_ids(graph)?;
    let mut old_to_new = HashMap::with_capacity(old_ids.len());
    let mut builder = GraphBuilder::new();

    for old_id in old_ids {
        let old_op = graph.node(old_id)?.op();
        let mapped_old_op = map_op(old_id, old_op);
        let remapped_op = mapped_old_op
            .map_operands(|operand| old_to_new.get(&operand).copied().unwrap_or(operand));
        let new_id = emit_op(&mut builder, &remapped_op, &HashMap::new())?;
        old_to_new.insert(old_id, new_id);
    }

    for &output in graph.outputs() {
        let mapped_old_output = map_output(output);
        let new_output = map_old_id(&old_to_new, mapped_old_output)?;
        builder.mark_output(new_output)?;
    }

    Ok(builder.build()?)
}

fn emit_op(
    builder: &mut GraphBuilder,
    op: &Op,
    old_to_new: &HashMap<NodeId, NodeId>,
) -> Result<NodeId, SurgeryError> {
    let id = match op {
        Op::Input { name, ty } => builder.input(name.clone(), ty.dtype(), ty.shape().clone())?,
        Op::Const { data, shape } => builder.constant(data.clone(), shape.clone())?,
        Op::Add(left, right) => builder.add(
            map_if_present(old_to_new, *left),
            map_if_present(old_to_new, *right),
        )?,
        Op::Sub(left, right) => builder.sub(
            map_if_present(old_to_new, *left),
            map_if_present(old_to_new, *right),
        )?,
        Op::Mul(left, right) => builder.mul(
            map_if_present(old_to_new, *left),
            map_if_present(old_to_new, *right),
        )?,
        Op::Neg(input) => builder.neg(map_if_present(old_to_new, *input))?,
        Op::Relu(input) => builder.relu(map_if_present(old_to_new, *input))?,
        Op::Matmul(left, right) => builder.matmul(
            map_if_present(old_to_new, *left),
            map_if_present(old_to_new, *right),
        )?,
        Op::Sum {
            input,
            axes,
            keepdim,
        } => builder.sum(map_if_present(old_to_new, *input), axes.clone(), *keepdim)?,
        Op::Reshape { input, new_shape } => {
            builder.reshape(map_if_present(old_to_new, *input), new_shape.clone())?
        }
    };
    Ok(id)
}

fn sorted_node_ids(graph: &Graph) -> Result<Vec<NodeId>, SurgeryError> {
    let mut ids = graph.topo_order()?;
    ids.sort_by_key(|id| id.index());
    Ok(ids)
}

fn collect_reachable(
    graph: &Graph,
    node: NodeId,
    keep: &mut HashSet<NodeId>,
) -> Result<(), SurgeryError> {
    if !keep.insert(node) {
        return Ok(());
    }
    for operand in graph.node(node)?.op().operands() {
        collect_reachable(graph, operand, keep)?;
    }
    Ok(())
}

fn validate_node_id(graph: &Graph, id: NodeId) -> Result<(), SurgeryError> {
    graph
        .node(id)
        .map(|_| ())
        .map_err(|_| SurgeryError::InvalidNodeId { id })
}

fn map_old_id(
    old_to_new: &HashMap<NodeId, NodeId>,
    old_id: NodeId,
) -> Result<NodeId, SurgeryError> {
    old_to_new
        .get(&old_id)
        .copied()
        .ok_or(SurgeryError::InvalidNodeId { id: old_id })
}

fn map_if_present(old_to_new: &HashMap<NodeId, NodeId>, id: NodeId) -> NodeId {
    old_to_new.get(&id).copied().unwrap_or(id)
}
