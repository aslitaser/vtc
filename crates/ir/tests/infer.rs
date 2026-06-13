//! Integration tests for graph type inference.

use proptest::prelude::{prop_assert_eq, proptest};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorType, TypeError, infer_types};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn input(builder: &mut GraphBuilder, name: &str, dtype: DType, dims: &[usize]) -> NodeId {
    builder
        .input(name, dtype, shape(dims))
        .expect("input construction is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> vtc_ir::Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output id is valid");
    builder.build().expect("graph is structurally valid")
}

fn assert_type(types: &vtc_ir::GraphTypes, id: NodeId, dtype: DType, dims: &[usize]) {
    let expected = TensorType::new(dtype, shape(dims));
    assert_eq!(types.type_of(id), Some(&expected));
}

fn assert_infer_error(graph: &Graph, expected: &TypeError) {
    assert_eq!(
        &infer_types(graph).expect_err("type inference should fail"),
        expected,
    );
}

#[test]
fn infers_matmul_output_shape() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[8, 4]);
    let w = input(&mut builder, "w", DType::F32, &[4, 2]);
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    let types = infer_types(&graph).expect("types infer");

    assert_type(&types, x, DType::F32, &[8, 4]);
    assert_type(&types, w, DType::F32, &[4, 2]);
    assert_type(&types, matmul, DType::F32, &[8, 2]);
    assert_eq!(
        types.output_types(&graph).expect("output types exist"),
        vec![&TensorType::new(DType::F32, shape(&[8, 2]))],
    );
}

#[test]
fn infers_elementwise_and_unary_types() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::F32, &[3, 3]);
    let b = input(&mut builder, "b", DType::F32, &[3, 3]);
    let add = builder.add(a, b).expect("add operands are valid");
    let neg = builder.neg(add).expect("neg operand is valid");
    let relu = builder.relu(neg).expect("relu operand is valid");
    let graph = finish(builder, relu);

    let types = infer_types(&graph).expect("types infer");

    assert_type(&types, add, DType::F32, &[3, 3]);
    assert_type(&types, neg, DType::F32, &[3, 3]);
    assert_type(&types, relu, DType::F32, &[3, 3]);
}

#[test]
fn infers_sum_shapes() {
    let mut remove_builder = GraphBuilder::new();
    let x = input(&mut remove_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = remove_builder
        .sum(x, vec![1], false)
        .expect("sum operand is valid");
    let graph = finish(remove_builder, sum);
    let types = infer_types(&graph).expect("types infer");
    assert_type(&types, sum, DType::F32, &[2, 4]);

    let mut keep_builder = GraphBuilder::new();
    let x = input(&mut keep_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = keep_builder
        .sum(x, vec![1], true)
        .expect("sum operand is valid");
    let graph = finish(keep_builder, sum);
    let types = infer_types(&graph).expect("types infer");
    assert_type(&types, sum, DType::F32, &[2, 1, 4]);

    let mut scalar_builder = GraphBuilder::new();
    let x = input(&mut scalar_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = scalar_builder
        .sum(x, vec![0, 1, 2], false)
        .expect("sum operand is valid");
    let graph = finish(scalar_builder, sum);
    let types = infer_types(&graph).expect("types infer");
    assert_type(&types, sum, DType::F32, &[]);

    let mut identity_builder = GraphBuilder::new();
    let x = input(&mut identity_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = identity_builder
        .sum(x, Vec::new(), false)
        .expect("sum operand is valid");
    let graph = finish(identity_builder, sum);
    let types = infer_types(&graph).expect("types infer");
    assert_type(&types, sum, DType::F32, &[2, 3, 4]);
}

#[test]
fn infers_reshape_shape() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[2, 6]);
    let reshape = builder
        .reshape(x, shape(&[3, 4]))
        .expect("reshape operand is valid");
    let graph = finish(builder, reshape);

    let types = infer_types(&graph).expect("types infer");

    assert_type(&types, reshape, DType::F32, &[3, 4]);
}

#[test]
fn infers_end_to_end_matmul_relu_sum() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[8, 4]);
    let w = input(&mut builder, "w", DType::F32, &[4, 2]);
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let relu = builder.relu(matmul).expect("relu operand is valid");
    let sum = builder
        .sum(relu, vec![1], false)
        .expect("sum operand is valid");
    let graph = finish(builder, sum);

    let types = infer_types(&graph).expect("types infer");

    assert_type(&types, x, DType::F32, &[8, 4]);
    assert_type(&types, w, DType::F32, &[4, 2]);
    assert_type(&types, matmul, DType::F32, &[8, 2]);
    assert_type(&types, relu, DType::F32, &[8, 2]);
    assert_type(&types, sum, DType::F32, &[8]);
}

#[test]
fn rejects_elementwise_shape_mismatch() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::F32, &[3, 3]);
    let b = input(&mut builder, "b", DType::F32, &[3, 4]);
    let add = builder.add(a, b).expect("add operands are valid");
    let graph = finish(builder, add);

    assert_infer_error(
        &graph,
        &TypeError::ShapeMismatch {
            op: "add",
            lhs: shape(&[3, 3]),
            rhs: shape(&[3, 4]),
        },
    );
}

#[test]
fn rejects_elementwise_dtype_mismatch() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::F32, &[3, 3]);
    let b = input(&mut builder, "b", DType::I32, &[3, 3]);
    let add = builder.add(a, b).expect("add operands are valid");
    let graph = finish(builder, add);

    assert_infer_error(
        &graph,
        &TypeError::DTypeMismatch {
            op: "add",
            lhs: DType::F32,
            rhs: DType::I32,
        },
    );
}

#[test]
fn rejects_matmul_contraction_mismatch() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[8, 4]);
    let w = input(&mut builder, "w", DType::F32, &[5, 2]);
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    assert_infer_error(
        &graph,
        &TypeError::MatmulContraction {
            lhs: shape(&[8, 4]),
            rhs: shape(&[5, 2]),
        },
    );
}

#[test]
fn rejects_matmul_rank_mismatch() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[2, 3, 4]);
    let w = input(&mut builder, "w", DType::F32, &[4, 2]);
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    assert_infer_error(
        &graph,
        &TypeError::MatmulRank {
            lhs_rank: 3,
            rhs_rank: 2,
        },
    );
}

#[test]
fn rejects_matmul_dtype_mismatch() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[8, 4]);
    let w = input(&mut builder, "w", DType::I32, &[4, 2]);
    let matmul = builder.matmul(x, w).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    assert_infer_error(
        &graph,
        &TypeError::DTypeMismatch {
            op: "matmul",
            lhs: DType::F32,
            rhs: DType::I32,
        },
    );
}

#[test]
fn rejects_bad_sum_axes() {
    let mut out_of_range_builder = GraphBuilder::new();
    let x = input(&mut out_of_range_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = out_of_range_builder
        .sum(x, vec![3], false)
        .expect("sum operand is valid");
    let graph = finish(out_of_range_builder, sum);

    assert_infer_error(&graph, &TypeError::SumAxisOutOfRange { axis: 3, rank: 3 });

    let mut duplicate_builder = GraphBuilder::new();
    let x = input(&mut duplicate_builder, "x", DType::F32, &[2, 3, 4]);
    let sum = duplicate_builder
        .sum(x, vec![1, 1], false)
        .expect("sum operand is valid");
    let graph = finish(duplicate_builder, sum);

    assert_infer_error(&graph, &TypeError::SumDuplicateAxis { axis: 1 });
}

#[test]
fn rejects_reshape_numel_mismatch() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[2, 6]);
    let reshape = builder
        .reshape(x, shape(&[4, 4]))
        .expect("reshape operand is valid");
    let graph = finish(builder, reshape);

    assert_infer_error(
        &graph,
        &TypeError::ReshapeNumelMismatch { from: 12, to: 16 },
    );
}

#[test]
fn rejects_bool_relu_and_add() {
    let mut relu_builder = GraphBuilder::new();
    let x = input(&mut relu_builder, "x", DType::Bool, &[3, 3]);
    let relu = relu_builder.relu(x).expect("relu operand is valid");
    let graph = finish(relu_builder, relu);

    assert_infer_error(
        &graph,
        &TypeError::NonNumericDType {
            op: "relu",
            dtype: DType::Bool,
        },
    );

    let mut add_builder = GraphBuilder::new();
    let a = input(&mut add_builder, "a", DType::Bool, &[3, 3]);
    let b = input(&mut add_builder, "b", DType::Bool, &[3, 3]);
    let add = add_builder.add(a, b).expect("add operands are valid");
    let graph = finish(add_builder, add);

    assert_infer_error(
        &graph,
        &TypeError::NonNumericDType {
            op: "add",
            dtype: DType::Bool,
        },
    );
}

proptest! {
    #[test]
    fn infers_random_matmul_output_shape(m in 1usize..16, k in 1usize..16, n in 1usize..16) {
        let mut builder = GraphBuilder::new();
        let x = input(&mut builder, "x", DType::F32, &[m, k]);
        let w = input(&mut builder, "w", DType::F32, &[k, n]);
        let matmul = builder.matmul(x, w).expect("matmul operands are valid");
        let graph = finish(builder, matmul);

        let types = infer_types(&graph).expect("types infer");
        let expected = TensorType::new(DType::F32, shape(&[m, n]));

        prop_assert_eq!(types.type_of(matmul), Some(&expected));
    }
}
