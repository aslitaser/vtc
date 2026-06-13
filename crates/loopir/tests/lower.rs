//! Integration tests for graph lowering and loop interpretation.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::{Tensor, eval};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape};
use vtc_loopir::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopError, ScalarExpr, Stmt,
    eval_loops, lower,
};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn tensor_i64(dims: &[usize], values: &[i64]) -> Tensor {
    Tensor::from_i64(shape(dims), values).expect("test tensor is valid")
}

fn input(builder: &mut GraphBuilder, name: &str, dtype: DType, dims: &[usize]) -> NodeId {
    builder
        .input(name, dtype, shape(dims))
        .expect("input is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn inputs(entries: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
    entries
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

fn assert_loop_matches_graph(graph: &Graph, inputs: &HashMap<String, Tensor>) -> Kernel {
    let kernel = lower(graph).expect("lowering succeeds");
    let graph_result = eval(graph, inputs).expect("graph evaluation succeeds");
    let loop_result = eval_loops(&kernel, inputs).expect("loop evaluation succeeds");
    assert_eq!(loop_result, graph_result);
    kernel
}

#[test]
fn lowers_elementwise_add_neg_relu() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[4]);
    let y = input(&mut builder, "y", DType::I64, &[4]);
    let add = builder.add(x, y).expect("add is valid");
    let neg = builder.neg(add).expect("neg is valid");
    let relu = builder.relu(neg).expect("relu is valid");
    let graph = finish(builder, relu);

    assert_loop_matches_graph(
        &graph,
        &inputs(&[
            ("x", tensor_i64(&[4], &[1, -4, 3, -8])),
            ("y", tensor_i64(&[4], &[2, 2, -10, 1])),
        ]),
    );
}

#[test]
fn matmul_known_example_matches_graph_oracle() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2, 2]);
    let b = input(&mut builder, "b", DType::I64, &[2, 2]);
    let matmul = builder.matmul(a, b).expect("matmul is valid");
    let graph = finish(builder, matmul);
    let values = inputs(&[
        ("a", tensor_i64(&[2, 2], &[1, 2, 3, 4])),
        ("b", tensor_i64(&[2, 2], &[5, 6, 7, 8])),
    ]);

    let kernel = lower(&graph).expect("lowering succeeds");
    let result = eval_loops(&kernel, &values).expect("loop evaluation succeeds");

    assert_eq!(result, vec![tensor_i64(&[2, 2], &[19, 22, 43, 50])]);
    assert_eq!(
        result,
        eval(&graph, &values).expect("graph evaluation succeeds")
    );
}

#[test]
fn sum_variants_match_graph_oracle() {
    let cases = [
        (vec![0], false),
        (vec![1], false),
        (vec![0, 1], false),
        (vec![0], true),
        (vec![1], true),
        (vec![0, 1], true),
        (Vec::new(), false),
    ];

    for (axes, keepdim) in cases {
        let mut builder = GraphBuilder::new();
        let x = input(&mut builder, "x", DType::I64, &[2, 3]);
        let sum = builder.sum(x, axes, keepdim).expect("sum is valid");
        let graph = finish(builder, sum);
        assert_loop_matches_graph(
            &graph,
            &inputs(&[("x", tensor_i64(&[2, 3], &[1, 2, 3, 4, 5, 6]))]),
        );
    }
}

#[test]
fn reshape_is_buffer_alias_and_evaluates_correctly() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2, 2]);
    let flat = builder.reshape(x, shape(&[4])).expect("reshape is valid");
    let graph = finish(builder, flat);

    let kernel = assert_loop_matches_graph(
        &graph,
        &inputs(&[("x", tensor_i64(&[2, 2], &[1, 2, 3, 4]))]),
    );

    assert_eq!(kernel.inputs(), &[BufferId::new(0)]);
    assert_eq!(kernel.outputs(), &[BufferId::new(0)]);
}

#[test]
fn composite_graph_matches_graph_oracle() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2, 2]);
    let b = input(&mut builder, "b", DType::I64, &[2, 2]);
    let w = input(&mut builder, "w", DType::I64, &[2, 2]);
    let add = builder.add(a, b).expect("add is valid");
    let matmul = builder.matmul(add, w).expect("matmul is valid");
    let relu = builder.relu(matmul).expect("relu is valid");
    let sum = builder.sum(relu, vec![1], false).expect("sum is valid");
    let graph = finish(builder, sum);

    assert_loop_matches_graph(
        &graph,
        &inputs(&[
            ("a", tensor_i64(&[2, 2], &[1, -2, 3, -4])),
            ("b", tensor_i64(&[2, 2], &[2, 5, -1, 1])),
            ("w", tensor_i64(&[2, 2], &[1, 2, 3, 4])),
        ]),
    );
}

#[test]
fn matmul_kernel_text_snapshot() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2, 2]);
    let b = input(&mut builder, "b", DType::I64, &[2, 2]);
    let matmul = builder.matmul(a, b).expect("matmul is valid");
    let graph = finish(builder, matmul);
    let kernel = lower(&graph).expect("lowering succeeds");

    assert_eq!(kernel.to_text(), MATMUL_TEXT);
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
            value: ScalarExpr::ConstScalar(BigRational::from_integer(BigInt::from(1))),
        }],
        Vec::new(),
        vec![BufferId::new(0)],
    );

    assert_eq!(
        eval_loops(&kernel, &HashMap::new()),
        Err(LoopError::IndexOutOfBounds {
            buffer: BufferId::new(0),
            index: 1,
            numel: 1,
        }),
    );
}

#[test]
fn random_graphs_match_graph_oracle() {
    let cfg = GenConfig {
        seed: 0x1500,
        ..GenConfig::default()
    };

    for iteration in 0..64 {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        assert_loop_matches_graph(&graph, &inputs);
    }
}

const MATMUL_TEXT: &str = "kernel {
  buffers:
    %b0 a[2, 2] role=input(\"a\")
    %b1 b[2, 2] role=input(\"b\")
    %b2 tmp[2, 2] role=temp
  body:
    for i0 in 0..2 {
      for i1 in 0..2 {
        %b2[2*i0 + i1] = 0
      }
    }
    for i0 in 0..2 {
      for i1 in 0..2 {
        for i2 in 0..2 {
          %b2[2*i0 + i1] = (%b2[2*i0 + i1] + (%b0[2*i0 + i2] * %b1[2*i2 + i1]))
        }
      }
    }
  outputs: %b2
}";
