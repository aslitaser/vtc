//! Initial bit-exact graph rewrite rules.

use vtc_ir::{Graph, NodeId, Op};

use crate::{Law, Rewrite, redefine_node, replace_all_uses};

/// Eliminates double negation, `-(-x) -> x`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NegNegElim;

impl Rewrite for NegNegElim {
    fn name(&self) -> &'static str {
        "neg-neg-elim"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::NegInvolutive]
    }

    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph> {
        let Op::Neg(inner) = graph.node(node).ok()?.op() else {
            return None;
        };
        let Op::Neg(input) = graph.node(*inner).ok()?.op() else {
            return None;
        };
        replace_all_uses(graph, node, *input).ok()
    }
}

/// Eliminates nested `ReLU`, `relu(relu(x)) -> relu(x)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReluIdempotentElim;

impl Rewrite for ReluIdempotentElim {
    fn name(&self) -> &'static str {
        "relu-idempotent-elim"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::ReluIdempotent]
    }

    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph> {
        let Op::Relu(inner) = graph.node(node).ok()?.op() else {
            return None;
        };
        let Op::Relu(_) = graph.node(*inner).ok()?.op() else {
            return None;
        };
        replace_all_uses(graph, node, *inner).ok()
    }
}

/// Fuses adjacent row-major reshapes, `reshape(reshape(x, s1), s2) -> reshape(x, s2)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReshapeReshapeFuse;

impl Rewrite for ReshapeReshapeFuse {
    fn name(&self) -> &'static str {
        "reshape-reshape-fuse"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::StructuralOnly]
    }

    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph> {
        let Op::Reshape {
            input: inner,
            new_shape,
        } = graph.node(node).ok()?.op()
        else {
            return None;
        };
        let Op::Reshape { input, .. } = graph.node(*inner).ok()?.op() else {
            return None;
        };
        redefine_node(
            graph,
            node,
            Op::Reshape {
                input: *input,
                new_shape: new_shape.clone(),
            },
        )
        .ok()
    }
}
