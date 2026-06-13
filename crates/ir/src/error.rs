//! Typed errors for graph IR construction and structural validation.

use thiserror::Error;

use crate::NodeId;

/// Errors produced by graph IR construction and structural validation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum IrError {
    /// A node id is outside the graph's node arena.
    #[error("invalid node id {id}; graph has {num_nodes} nodes")]
    InvalidNodeId {
        /// The invalid id.
        id: NodeId,
        /// The current number of nodes in the graph.
        num_nodes: usize,
    },

    /// A node references itself or a node later in build order.
    #[error("node {node} has forward or self reference to operand {operand}")]
    ForwardReference {
        /// The node containing the bad operand reference.
        node: NodeId,
        /// The operand that is not earlier in build order.
        operand: NodeId,
    },

    /// A graph input list entry does not point to an input node.
    #[error("node {0} is not an input")]
    NotAnInput(NodeId),

    /// The graph has no marked outputs.
    #[error("graph has no outputs")]
    NoOutputs,

    /// A constant payload length does not match its shape element count.
    #[error("constant data length mismatch: expected {expected}, got {got}")]
    ConstDataShapeMismatch {
        /// Expected element count from the constant shape.
        expected: usize,
        /// Actual payload element count.
        got: usize,
    },

    /// Shape element-count multiplication overflowed.
    #[error("shape element count overflow")]
    ShapeOverflow,

    /// The graph contains a cycle.
    #[error("graph contains a cycle")]
    Cycle,
}
