//! Integration tests for the graph IR public API.

use proptest::proptest;
use vtc_ir::{DType, Dim, GraphBuilder, IrError, Shape, TensorData};

fn matrix_shape() -> Shape {
    Shape::new(vec![Dim::new(2), Dim::new(2)])
}

fn assert_valid_topo_order(order: &[vtc_ir::NodeId], graph_node_count: usize) {
    assert_eq!(order.len(), graph_node_count);
    let mut seen = vec![false; graph_node_count];
    for &node in order {
        let index = node.index();
        assert!(index < graph_node_count);
        assert!(!seen[index]);
        seen[index] = true;
    }
}

#[test]
fn builds_elementwise_expression_graph() {
    let mut builder = GraphBuilder::new();
    let a = builder
        .input("a", DType::F32, matrix_shape())
        .expect("input a is valid");
    let b = builder
        .input("b", DType::F32, matrix_shape())
        .expect("input b is valid");
    let c = builder
        .input("c", DType::F32, matrix_shape())
        .expect("input c is valid");
    let add = builder.add(a, b).expect("add operands are valid");
    let mul = builder.mul(add, c).expect("mul operands are valid");
    builder.mark_output(mul).expect("output is valid");

    let graph = builder.build().expect("graph is structurally valid");
    let topo = graph.topo_order().expect("graph is a dag");

    assert_valid_topo_order(&topo, graph.num_nodes());
    assert_eq!(
        graph.to_text(),
        r#"graph {
  %0 = input "a" : f32[2, 2]
  %1 = input "b" : f32[2, 2]
  %2 = input "c" : f32[2, 2]
  %3 = add(%0, %1)
  %4 = mul(%3, %2)
  outputs: %4
}"#
    );
}

#[test]
fn builds_matmul_relu_graph() {
    let mut builder = GraphBuilder::new();
    let x = builder
        .input(
            "x",
            DType::F32,
            Shape::new(vec![Dim::new(128), Dim::new(256)]),
        )
        .expect("input x is valid");
    let w = builder
        .input(
            "w",
            DType::F32,
            Shape::new(vec![Dim::new(256), Dim::new(64)]),
        )
        .expect("input w is valid");
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let relu = builder.relu(matmul).expect("relu operand is valid");
    builder.mark_output(relu).expect("output is valid");

    let graph = builder.build().expect("graph is structurally valid");

    assert_eq!(
        graph.to_text(),
        r#"graph {
  %0 = input "x" : f32[128, 256]
  %1 = input "w" : f32[256, 64]
  %2 = matmul(%0, %1)
  %3 = relu(%2)
  outputs: %3
}"#
    );
}

#[test]
fn constant_rejects_data_shape_mismatch() {
    let mut builder = GraphBuilder::new();

    let result = builder.constant(
        TensorData::F32(vec![1.0, 2.0]),
        Shape::new(vec![Dim::new(3)]),
    );

    assert_eq!(
        result,
        Err(IrError::ConstDataShapeMismatch {
            expected: 3,
            got: 2,
        })
    );
}

proptest! {
    #[test]
    fn shape_numel_matches_checked_product(raw_dims in proptest::collection::vec(0usize..8, 0..8)) {
        let dims: Vec<_> = raw_dims.iter().copied().map(Dim::new).collect();
        let shape = Shape::new(dims);
        let expected = raw_dims
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim));

        assert_eq!(shape.rank(), raw_dims.len());
        assert_eq!(shape.numel(), expected.ok_or(IrError::ShapeOverflow));
    }
}
