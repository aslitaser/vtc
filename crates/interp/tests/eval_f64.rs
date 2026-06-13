//! Integration tests for IEEE `f64` graph evaluation.

use std::collections::HashMap;

use vtc_interp::{EvalError, TensorF64, eval_f64};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn tensor_f64(dims: &[usize], values: &[f64]) -> TensorF64 {
    TensorF64::from_f64(shape(dims), values).expect("test tensor is valid")
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

fn inputs(entries: &[(&str, TensorF64)]) -> HashMap<String, TensorF64> {
    entries
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect()
}

#[test]
fn evaluates_matmul_in_ascending_k_order() {
    let mut builder = GraphBuilder::new();
    let a = input(&mut builder, "a", DType::F64, &[2, 2]);
    let b = input(&mut builder, "b", DType::F64, &[2, 2]);
    let matmul = builder.matmul(a, b).expect("matmul operands are valid");
    let graph = finish(builder, matmul);

    let result = eval_f64(
        &graph,
        &inputs(&[
            ("a", tensor_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
            ("b", tensor_f64(&[2, 2], &[5.0, 6.0, 7.0, 8.0])),
        ]),
    )
    .expect("evaluation succeeds");

    let expected = tensor_f64(&[2, 2], &[19.0, 22.0, 43.0, 50.0]);
    assert!(result[0].bit_eq(&expected));
}

#[test]
fn sum_uses_order_sensitive_ascending_flat_order() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F64, &[3]);
    let sum = builder
        .sum(x, vec![0], false)
        .expect("sum operand is valid");
    let graph = finish(builder, sum);

    let result = eval_f64(
        &graph,
        &inputs(&[("x", tensor_f64(&[3], &[1.0e16, 1.0, -1.0e16]))]),
    )
    .expect("evaluation succeeds");

    assert_eq!(result[0].data()[0].to_bits(), 0.0f64.to_bits());
}

#[test]
fn relu_uses_comparison_semantics() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", DType::F64, &[4]);
    let relu = builder.relu(x).expect("relu operand is valid");
    let graph = finish(builder, relu);

    let result = eval_f64(
        &graph,
        &inputs(&[("x", tensor_f64(&[4], &[-0.0, f64::NAN, -2.0, 3.0]))]),
    )
    .expect("evaluation succeeds");

    let expected = tensor_f64(&[4], &[0.0, 0.0, 0.0, 3.0]);
    assert!(result[0].bit_eq(&expected));
}

#[test]
fn bit_eq_is_nan_payload_insensitive_and_signed_zero_sensitive() {
    assert!(!tensor_f64(&[1], &[0.0]).bit_eq(&tensor_f64(&[1], &[-0.0])));
    assert!(
        tensor_f64(&[1], &[f64::NAN])
            .bit_eq(&tensor_f64(&[1], &[f64::from_bits(0x7ff8_0000_0000_0001)]))
    );
    assert!(tensor_f64(&[1], &[1.5]).bit_eq(&tensor_f64(&[1], &[1.5])));
    assert!(!tensor_f64(&[1], &[1.5]).bit_eq(&tensor_f64(&[1], &[1.500_000_1])));
}

#[test]
fn from_data_lifts_supported_finite_constants() {
    let int_tensor = TensorF64::from_data(&TensorData::I64(vec![42, -7]), &shape(&[2]))
        .expect("integer constants lift");
    assert_eq!(int_tensor.data()[0].to_bits(), 42.0f64.to_bits());
    assert_eq!(int_tensor.data()[1].to_bits(), (-7.0f64).to_bits());

    let f32_tensor =
        TensorF64::from_data(&TensorData::F32(vec![0.5]), &shape(&[1])).expect("f32 lifts");
    assert_eq!(f32_tensor.data()[0].to_bits(), 0.5f64.to_bits());
}

#[test]
fn from_data_rejects_bool_and_non_finite_constants() {
    assert_eq!(
        TensorF64::from_data(&TensorData::Bool(vec![true]), &shape(&[1]))
            .expect_err("bool constants are unsupported"),
        EvalError::UnsupportedDtype { dtype: DType::Bool },
    );
    assert_eq!(
        TensorF64::from_data(&TensorData::F64(vec![f64::INFINITY]), &shape(&[1]))
            .expect_err("non-finite constants are unsupported"),
        EvalError::NonFiniteFloat,
    );
}
