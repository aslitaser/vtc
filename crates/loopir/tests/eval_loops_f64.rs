//! Integration tests for the IEEE `f64` loop interpreter.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use vtc_interp::eval_f64;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape};
use vtc_loopir::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopError, ScalarExpr, Stmt,
    TensorF64, eval_loops_f64, lower,
};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn scalar_shape() -> Shape {
    Shape::new(Vec::new())
}

fn tensor_f64(dims: &[usize], values: &[f64]) -> TensorF64 {
    TensorF64::from_f64(shape(dims), values).expect("test f64 tensor is valid")
}

fn scalar_f64(value: f64) -> TensorF64 {
    TensorF64::from_f64(scalar_shape(), &[value]).expect("test scalar tensor is valid")
}

fn input(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> NodeId {
    builder
        .input(name, DType::F64, shape(dims))
        .expect("input is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn inputs(entries: &[(&str, TensorF64)]) -> HashMap<String, TensorF64> {
    entries
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

fn assert_bit_eq(left: &[TensorF64], right: &[TensorF64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!(left.bit_eq(right), "left={left:?}, right={right:?}");
    }
}

fn assert_loop_matches_graph(graph: &Graph, values: &HashMap<String, TensorF64>) {
    let kernel = lower(graph).expect("lowering succeeds");
    let loop_result = eval_loops_f64(&kernel, values).expect("loop f64 evaluation succeeds");
    let graph_result = eval_f64(graph, values).expect("graph f64 evaluation succeeds");
    assert_bit_eq(&loop_result, &graph_result);
}

#[test]
fn lowered_graphs_match_graph_f64_oracle_bit_for_bit() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", &[2, 3]);
    let y = input(&mut builder, "y", &[2, 3]);
    let add = builder.add(x, y).expect("add is valid");
    let neg = builder.neg(add).expect("neg is valid");
    let relu = builder.relu(neg).expect("relu is valid");
    let graph = finish(builder, relu);
    assert_loop_matches_graph(
        &graph,
        &inputs(&[
            ("x", tensor_f64(&[2, 3], &[1.0, -2.0, 3.5, -0.0, 1e16, 4.0])),
            (
                "y",
                tensor_f64(&[2, 3], &[2.0, 2.0, -1.5, 0.0, -1e16, -8.0]),
            ),
        ]),
    );

    let mut builder = GraphBuilder::new();
    let lhs = input(&mut builder, "lhs", &[2, 3]);
    let rhs = input(&mut builder, "rhs", &[3, 2]);
    let matmul = builder.matmul(lhs, rhs).expect("matmul is valid");
    let graph = finish(builder, matmul);
    assert_loop_matches_graph(
        &graph,
        &inputs(&[
            ("lhs", tensor_f64(&[2, 3], &[1.0, 2.0, 3.0, -4.0, 5.0, 6.0])),
            (
                "rhs",
                tensor_f64(&[3, 2], &[7.0, -8.0, 9.0, 10.0, -11.0, 12.0]),
            ),
        ]),
    );

    for (axes, keepdim) in [
        (vec![0], false),
        (vec![1], false),
        (vec![0, 1], false),
        (vec![0], true),
        (vec![1], true),
        (vec![0, 1], true),
    ] {
        let mut builder = GraphBuilder::new();
        let values = input(&mut builder, "values", &[2, 3]);
        let sum = builder.sum(values, axes, keepdim).expect("sum is valid");
        let graph = finish(builder, sum);
        assert_loop_matches_graph(
            &graph,
            &inputs(&[(
                "values",
                tensor_f64(&[2, 3], &[1e16, -1e16, 1.0, -2.0, 3.0, 4.0]),
            )]),
        );
    }

    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", &[2, 2]);
    let b = input(&mut builder, "b", &[2, 2]);
    let weights = input(&mut builder, "weights", &[2, 2]);
    let add = builder.add(a, b).expect("add is valid");
    let matmul = builder.matmul(add, weights).expect("matmul is valid");
    let relu = builder.relu(matmul).expect("relu is valid");
    let sum = builder.sum(relu, vec![1], false).expect("sum is valid");
    let graph = finish(builder, sum);
    assert_loop_matches_graph(
        &graph,
        &inputs(&[
            ("a", tensor_f64(&[2, 2], &[1.0, -2.0, 3.0, -4.0])),
            ("b", tensor_f64(&[2, 2], &[2.0, 5.0, -1.0, 1.0])),
            ("weights", tensor_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        ]),
    );
}

#[test]
fn sum_uses_kernel_order_and_preserves_rounding_loss() {
    let mut builder = GraphBuilder::new();
    let values = input(&mut builder, "values", &[3]);
    let sum = builder.sum(values, vec![0], false).expect("sum is valid");
    let graph = finish(builder, sum);
    let kernel = lower(&graph).expect("lowering succeeds");
    let result = eval_loops_f64(
        &kernel,
        &inputs(&[("values", tensor_f64(&[3], &[1e16, 1.0, -1e16]))]),
    )
    .expect("loop f64 evaluation succeeds");

    assert!(result[0].bit_eq(&scalar_f64(0.0)));
}

#[test]
fn relu_uses_comparison_semantics() {
    let mut builder = GraphBuilder::new();
    let values = input(&mut builder, "values", &[4]);
    let relu = builder.relu(values).expect("relu is valid");
    let graph = finish(builder, relu);
    let kernel = lower(&graph).expect("lowering succeeds");
    let result = eval_loops_f64(
        &kernel,
        &inputs(&[("values", tensor_f64(&[4], &[-0.0, f64::NAN, -2.0, 3.0]))]),
    )
    .expect("loop f64 evaluation succeeds");

    assert!(result[0].bit_eq(&tensor_f64(&[4], &[0.0, 0.0, 0.0, 3.0])));
}

#[test]
fn out_of_bounds_access_returns_error() {
    let buffer = Buffer::new(
        BufferId::new(0),
        "tmp".to_owned(),
        shape(&[1]),
        BufferRole::Temp,
    );
    let kernel = Kernel::new(
        vec![buffer],
        vec![Stmt::Assign {
            target: BufferRef {
                buffer: BufferId::new(0),
                index: AffineExpr::constant(1),
            },
            value: ScalarExpr::ConstScalar(BigRational::from_integer(BigInt::from(0))),
        }],
        Vec::new(),
        vec![BufferId::new(0)],
    );

    let error = eval_loops_f64(&kernel, &HashMap::new()).expect_err("evaluation should fail");
    assert_eq!(
        error,
        LoopError::IndexOutOfBounds {
            buffer: BufferId::new(0),
            index: 1,
            numel: 1,
        },
    );
}
