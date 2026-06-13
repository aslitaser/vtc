//! Integration tests for exact-rational graph evaluation.

use std::collections::HashMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use proptest::prelude::{prop_assert_eq, proptest};
use vtc_interp::{EvalError, Tensor, eval};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData, TypeError};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn br(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn tensor_i64(dims: &[usize], values: &[i64]) -> Tensor {
    Tensor::from_i64(shape(dims), values).expect("test tensor is valid")
}

fn tensor_f64(dims: &[usize], values: &[f64]) -> Tensor {
    Tensor::from_f64(shape(dims), values).expect("test tensor is valid")
}

fn input(builder: &mut GraphBuilder, name: &str, dtype: DType, dims: &[usize]) -> NodeId {
    builder
        .input(name, dtype, shape(dims))
        .expect("input construction is valid")
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

#[test]
fn evaluates_matmul_exactly() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2, 2]);
    let b = input(&mut builder, "b", DType::I64, &[2, 2]);
    let matmul = builder.matmul(a, b).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    let result = eval(
        &graph,
        &inputs(&[
            ("a", tensor_i64(&[2, 2], &[1, 2, 3, 4])),
            ("b", tensor_i64(&[2, 2], &[5, 6, 7, 8])),
        ]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result, vec![tensor_i64(&[2, 2], &[19, 22, 43, 50])]);
}

#[test]
fn evaluates_elementwise_ops_exactly() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2]);
    let b = input(&mut builder, "b", DType::I64, &[2]);
    let add = builder.add(a, b).expect("add operands are valid");
    let sub = builder.sub(add, b).expect("sub operands are valid");
    let mul = builder.mul(sub, a).expect("mul operands are valid");
    let neg = builder.neg(mul).expect("neg operand is valid");
    let graph = finish(builder, neg);

    let result = eval(
        &graph,
        &inputs(&[
            ("a", tensor_i64(&[2], &[2, 3])),
            ("b", tensor_i64(&[2], &[5, 7])),
        ]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result, vec![tensor_i64(&[2], &[-4, -9])]);
}

#[test]
fn evaluates_relu_exactly() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F64, &[4]);
    let relu = builder.relu(x).expect("relu operand is valid");
    let graph = finish(builder, relu);

    let result = eval(
        &graph,
        &inputs(&[("x", tensor_f64(&[4], &[-2.0, -0.5, 0.0, 1.5]))]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result[0].data(), &[br(0, 1), br(0, 1), br(0, 1), br(3, 2)]);
}

#[test]
fn lifts_f64_without_decimal_rounding() {
    let half = Tensor::from_f64(shape(&[1]), &[0.5]).expect("finite float lifts");
    assert_eq!(half.data(), &[br(1, 2)]);

    let one_tenth_float = Tensor::from_f64(shape(&[1]), &[0.1]).expect("finite float lifts");
    assert_ne!(one_tenth_float.data(), &[br(1, 10)]);
}

#[test]
fn evaluates_sum_variants_exactly() {
    let cases = [
        (vec![0], false, shape(&[3]), vec![5, 7, 9]),
        (vec![1], false, shape(&[2]), vec![6, 15]),
        (vec![0, 1], false, shape(&[]), vec![21]),
        (vec![0], true, shape(&[1, 3]), vec![5, 7, 9]),
        (vec![1], true, shape(&[2, 1]), vec![6, 15]),
        (vec![0, 1], true, shape(&[1, 1]), vec![21]),
        (Vec::new(), false, shape(&[2, 3]), vec![1, 2, 3, 4, 5, 6]),
    ];

    for (axes, keepdim, expected_shape, expected_values) in cases {
        let mut builder = GraphBuilder::new();
        let x = input(&mut builder, "x", DType::F32, &[2, 3]);
        let sum = builder.sum(x, axes, keepdim).expect("sum operand is valid");
        let graph = finish(builder, sum);

        let result = eval(
            &graph,
            &inputs(&[("x", tensor_i64(&[2, 3], &[1, 2, 3, 4, 5, 6]))]),
        )
        .expect("evaluation succeeds");

        assert_eq!(result[0].shape(), &expected_shape);
        assert_eq!(
            result[0].data(),
            Tensor::from_i64(expected_shape, &expected_values)
                .expect("expected tensor is valid")
                .data(),
        );
    }
}

#[test]
fn evaluates_reshape_as_metadata_change() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F32, &[2, 2]);
    let flat = builder.reshape(x, shape(&[4])).expect("reshape is valid");
    let wide = builder
        .reshape(flat, shape(&[1, 4]))
        .expect("reshape is valid");
    let graph = finish(builder, wide);

    let result = eval(
        &graph,
        &inputs(&[("x", tensor_i64(&[2, 2], &[1, 2, 3, 4]))]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result[0].shape(), &shape(&[1, 4]));
    assert_eq!(result[0].data(), tensor_i64(&[4], &[1, 2, 3, 4]).data());
}

#[test]
fn evaluates_end_to_end_graph_exactly() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2, 2]);
    let b = input(&mut builder, "b", DType::I64, &[2, 2]);
    let w = input(&mut builder, "w", DType::I64, &[2, 2]);
    let add = builder.add(a, b).expect("add operands are valid");
    let matmul = builder.matmul(add, w).expect("matmul operands are valid");
    let relu = builder.relu(matmul).expect("relu operand is valid");
    let sum = builder
        .sum(relu, vec![1], false)
        .expect("sum operand is valid");
    let graph = finish(builder, sum);

    let result = eval(
        &graph,
        &inputs(&[
            ("a", tensor_i64(&[2, 2], &[1, 2, 3, 4])),
            ("b", tensor_i64(&[2, 2], &[1, 1, 1, 1])),
            ("w", tensor_i64(&[2, 2], &[1, 0, 0, 1])),
        ]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result, vec![tensor_i64(&[2], &[5, 9])]);
}

#[test]
fn reports_missing_input() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[1]);
    let graph = finish(builder, x);

    assert_eq!(
        eval(&graph, &HashMap::new()),
        Err(EvalError::MissingInput("x".to_owned())),
    );
}

#[test]
fn reports_input_shape_mismatch() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::I64, &[2]);
    let graph = finish(builder, x);

    assert_eq!(
        eval(&graph, &inputs(&[("x", tensor_i64(&[3], &[1, 2, 3]))])),
        Err(EvalError::InputShapeMismatch {
            name: "x".to_owned(),
            expected: shape(&[2]),
            got: shape(&[3]),
        }),
    );
}

#[test]
fn rejects_bool_values() {
    let mut input_builder = GraphBuilder::new();
    let x = input(&mut input_builder, "x", DType::Bool, &[1]);
    let graph = finish(input_builder, x);

    assert_eq!(
        eval(&graph, &inputs(&[("x", tensor_i64(&[1], &[1]))])),
        Err(EvalError::UnsupportedDtype { dtype: DType::Bool }),
    );

    let mut const_builder = GraphBuilder::new();
    let c = const_builder
        .constant(TensorData::Bool(vec![true]), shape(&[1]))
        .expect("const is structurally valid");
    let graph = finish(const_builder, c);

    assert_eq!(
        eval(&graph, &HashMap::new()),
        Err(EvalError::UnsupportedDtype { dtype: DType::Bool }),
    );
}

#[test]
fn refuses_ill_typed_graph() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::I64, &[2]);
    let b = input(&mut builder, "b", DType::I64, &[3]);
    let add = builder
        .add(a, b)
        .expect("add operands are structurally valid");
    let graph = finish(builder, add);

    assert!(matches!(
        eval(
            &graph,
            &inputs(&[
                ("a", tensor_i64(&[2], &[1, 2])),
                ("b", tensor_i64(&[3], &[1, 2, 3]))
            ])
        ),
        Err(EvalError::Type(TypeError::ShapeMismatch { .. })),
    ));
}

#[test]
fn rejects_non_finite_constants() {
    let mut builder = GraphBuilder::new();
    let c = builder
        .constant(TensorData::F64(vec![f64::NAN, f64::INFINITY]), shape(&[2]))
        .expect("const is structurally valid");
    let graph = finish(builder, c);

    assert_eq!(
        eval(&graph, &HashMap::new()),
        Err(EvalError::NonFiniteFloat)
    );
}

proptest! {
    #[test]
    fn evaluates_random_small_int_matmul(
        m in 1usize..4,
        k in 1usize..4,
        n in 1usize..4,
        a_pool in proptest::collection::vec(-5i64..5, 16),
        b_pool in proptest::collection::vec(-5i64..5, 16),
    ) {
        let a_len = m * k;
        let b_len = k * n;
        let a_values = &a_pool[..a_len];
        let b_values = &b_pool[..b_len];

        let mut builder = GraphBuilder::new();
        let a = input(&mut builder, "a", DType::I64, &[m, k]);
        let b = input(&mut builder, "b", DType::I64, &[k, n]);
        let matmul = builder.matmul(a, b).expect("matmul operands are valid");
        let graph = finish(builder, matmul);

        let result = eval(
            &graph,
            &inputs(&[
                ("a", tensor_i64(&[m, k], a_values)),
                ("b", tensor_i64(&[k, n], b_values)),
            ]),
        )
        .expect("evaluation succeeds");

        let mut expected = vec![BigInt::zero(); m * n];
        for row in 0..m {
            for col in 0..n {
                for inner in 0..k {
                    expected[row * n + col] +=
                        BigInt::from(a_values[row * k + inner])
                            * BigInt::from(b_values[inner * n + col]);
                }
            }
        }
        let expected = expected
            .into_iter()
            .map(BigRational::from_integer)
            .collect::<Vec<_>>();

        prop_assert_eq!(result[0].shape(), &shape(&[m, n]));
        prop_assert_eq!(result[0].data(), expected.as_slice());
    }
}
