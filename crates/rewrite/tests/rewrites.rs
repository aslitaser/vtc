//! Tests for graph surgery and the first concrete rewrites.

use std::collections::HashMap;

use vtc_interp::{Tensor, eval};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorType, infer_types};
use vtc_rewrite::{
    Law, NegNegElim, NumericSafety, ReluIdempotentElim, ReshapeReshapeFuse, Rewrite, RewriteMode,
    RuleSet, SurgeryError, prune_and_rebuild, redefine_node, replace_all_uses,
};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn input(builder: &mut GraphBuilder, name: &str, dtype: DType, dims: &[usize]) -> NodeId {
    builder
        .input(name, dtype, shape(dims))
        .expect("input is structurally valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output id is valid");
    builder.build().expect("graph is structurally valid")
}

fn inputs(entries: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
    entries
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

fn tensor_i64(dims: &[usize], values: &[i64]) -> Tensor {
    Tensor::from_i64(shape(dims), values).expect("test tensor is valid")
}

fn output_type(graph: &Graph) -> TensorType {
    let types = infer_types(graph).expect("graph is well typed");
    types
        .output_types(graph)
        .expect("output type exists")
        .into_iter()
        .next()
        .expect("graph has one output")
        .clone()
}

fn assert_equivalent(original: &Graph, rewritten: &Graph, inputs: &HashMap<String, Tensor>) {
    assert_eq!(
        eval(original, inputs).expect("original evaluates"),
        eval(rewritten, inputs).expect("rewritten evaluates"),
    );
    assert_eq!(output_type(original), output_type(rewritten));
}

fn assert_surgery_error(result: Result<Graph, SurgeryError>, expected: &SurgeryError) {
    assert_eq!(&result.expect_err("surgery should fail"), expected);
}

fn invalid_id() -> NodeId {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[1]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let _graph = finish(builder, neg);
    neg
}

#[test]
fn neg_neg_elim_is_equivalent_and_prunable() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[4]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let double_neg = builder.neg(neg).expect("neg operand is valid");
    let graph = finish(builder, double_neg);
    let inputs = inputs(&[("x", tensor_i64(&[4], &[1, -2, 3, -4]))]);

    let rewritten = NegNegElim
        .try_at(&graph, double_neg)
        .expect("pattern matches");
    assert_equivalent(&graph, &rewritten, &inputs);

    let pruned = prune_and_rebuild(&rewritten).expect("prune succeeds");
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(pruned.num_nodes(), 1);
    assert_equivalent(&graph, &pruned, &inputs);
}

#[test]
fn relu_idempotent_elim_is_equivalent_and_prunable() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[4]);
    let relu = builder.relu(x).expect("relu operand is valid");
    let double_relu = builder.relu(relu).expect("relu operand is valid");
    let graph = finish(builder, double_relu);
    let inputs = inputs(&[("x", tensor_i64(&[4], &[1, -2, 0, 3]))]);

    let rewritten = ReluIdempotentElim
        .try_at(&graph, double_relu)
        .expect("pattern matches");
    assert_equivalent(&graph, &rewritten, &inputs);

    let pruned = prune_and_rebuild(&rewritten).expect("prune succeeds");
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(pruned.num_nodes(), 2);
    assert_equivalent(&graph, &pruned, &inputs);
}

#[test]
fn reshape_reshape_fuse_is_equivalent_and_prunable() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2, 2]);
    let flat = builder.reshape(x, shape(&[4])).expect("reshape is valid");
    let wide = builder
        .reshape(flat, shape(&[1, 4]))
        .expect("reshape is valid");
    let graph = finish(builder, wide);
    let inputs = inputs(&[("x", tensor_i64(&[2, 2], &[1, 2, 3, 4]))]);

    let rewritten = ReshapeReshapeFuse
        .try_at(&graph, wide)
        .expect("pattern matches");
    assert_equivalent(&graph, &rewritten, &inputs);

    let pruned = prune_and_rebuild(&rewritten).expect("prune succeeds");
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(pruned.num_nodes(), 2);
    assert_equivalent(&graph, &pruned, &inputs);
}

#[test]
fn rewrites_return_none_without_matching_pattern() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let relu = builder.relu(x).expect("relu operand is valid");
    let reshape = builder.reshape(x, shape(&[2])).expect("reshape is valid");
    let graph = finish(builder, reshape);

    assert!(NegNegElim.try_at(&graph, neg).is_none());
    assert!(NegNegElim.try_at(&graph, relu).is_none());
    assert!(ReluIdempotentElim.try_at(&graph, relu).is_none());
    assert!(ReluIdempotentElim.try_at(&graph, neg).is_none());
    assert!(ReshapeReshapeFuse.try_at(&graph, reshape).is_none());
    assert!(ReshapeReshapeFuse.try_at(&graph, neg).is_none());
}

#[test]
fn replace_all_uses_redirects_operands_and_outputs() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let relu = builder.relu(neg).expect("relu operand is valid");
    let graph = finish(builder, relu);

    let rewritten = replace_all_uses(&graph, neg, x).expect("replacement succeeds");

    assert!(rewritten.validate_structure().is_ok());
    assert!(rewritten.to_text().contains("%2 = relu(%0)"));

    let output_rewritten = replace_all_uses(&graph, relu, neg).expect("replacement succeeds");
    assert_eq!(output_rewritten.outputs(), &[neg]);
}

#[test]
fn replace_all_uses_rejects_invalid_ids() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[1]);
    let graph = finish(builder, x);
    let bad = invalid_id();

    assert_surgery_error(
        replace_all_uses(&graph, bad, x),
        &SurgeryError::InvalidNodeId { id: bad },
    );
    assert_surgery_error(
        replace_all_uses(&graph, x, bad),
        &SurgeryError::InvalidNodeId { id: bad },
    );
}

#[test]
fn redefine_node_changes_op_and_rejects_forward_references() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[1]);
    let y = input(&mut builder, "y", DType::I64, &[1]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let graph = finish(builder, neg);

    let rewritten = redefine_node(&graph, neg, Op::Relu(x)).expect("redefine succeeds");
    assert!(rewritten.to_text().contains("%2 = relu(%0)"));

    assert_surgery_error(
        redefine_node(&graph, y, Op::Neg(neg)),
        &SurgeryError::ForwardReference {
            node: y,
            operand: neg,
        },
    );
}

#[test]
fn prune_drops_dead_nodes_keeps_all_inputs_and_preserves_eval() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2]);
    let _unused = input(&mut builder, "unused", DType::I64, &[2]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let _dead = builder.relu(neg).expect("relu operand is valid");
    let graph = finish(builder, neg);
    let inputs = inputs(&[
        ("x", tensor_i64(&[2], &[1, -2])),
        ("unused", tensor_i64(&[2], &[5, 6])),
    ]);

    let pruned = prune_and_rebuild(&graph).expect("prune succeeds");

    assert_eq!(graph.num_nodes(), 4);
    assert_eq!(pruned.num_nodes(), 3);
    assert_eq!(pruned.inputs().len(), 2);
    assert!(pruned.validate_structure().is_ok());
    assert_equivalent(&graph, &pruned, &inputs);
}

#[test]
fn combined_rewrites_preserve_eval_and_reduce_nodes() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2, 2]);
    let neg = builder.neg(x).expect("neg operand is valid");
    let double_neg = builder.neg(neg).expect("neg operand is valid");
    let flat = builder
        .reshape(double_neg, shape(&[4]))
        .expect("reshape is valid");
    let wide = builder
        .reshape(flat, shape(&[1, 4]))
        .expect("reshape is valid");
    let graph = finish(builder, wide);
    let inputs = inputs(&[("x", tensor_i64(&[2, 2], &[1, -2, 3, -4]))]);

    let rewritten = NegNegElim
        .try_at(&graph, double_neg)
        .expect("neg-neg matches");
    let rewritten = ReshapeReshapeFuse
        .try_at(&rewritten, wide)
        .expect("reshape-reshape matches");
    let pruned = prune_and_rebuild(&rewritten).expect("prune succeeds");

    assert_equivalent(&graph, &pruned, &inputs);
    assert_eq!(graph.num_nodes(), 5);
    assert_eq!(pruned.num_nodes(), 2);
}

#[test]
fn concrete_rewrites_are_bit_exact_and_strict_enabled() {
    let rules: Vec<Box<dyn Rewrite>> = vec![
        Box::new(NegNegElim),
        Box::new(ReluIdempotentElim),
        Box::new(ReshapeReshapeFuse),
    ];
    for rule in &rules {
        assert_eq!(rule.safety(), NumericSafety::BitExact);
    }

    let mut ruleset = RuleSet::new();
    for rule in rules {
        ruleset.add(rule);
    }

    let strict_names = ruleset
        .enabled(RewriteMode::Strict)
        .into_iter()
        .map(Rewrite::name)
        .collect::<Vec<_>>();
    assert_eq!(
        strict_names,
        vec![
            "neg-neg-elim",
            "relu-idempotent-elim",
            "reshape-reshape-fuse",
        ],
    );
}

#[test]
fn concrete_rewrites_declare_expected_laws() {
    assert_eq!(NegNegElim.laws(), &[Law::NegInvolutive]);
    assert_eq!(ReluIdempotentElim.laws(), &[Law::ReluIdempotent]);
    assert_eq!(ReshapeReshapeFuse.laws(), &[Law::StructuralOnly]);
}
