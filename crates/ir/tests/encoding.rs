//! Tests for the source graph format shared with Lean.

use serde_json::Value;
use vtc_ir::{DType, Dim, GraphBuilder, GraphEncodeError, Shape, TensorData, encode_graph_v1};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

#[test]
fn encoder_matches_shared_golden_fixture() {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input("features\"\\\n雪", DType::F64, shape(&[2]))
        .expect("input is valid");
    let constant = builder
        .constant(TensorData::F64(vec![-0.0, -2.0]), shape(&[2]))
        .expect("constant is valid");
    let sum = builder.add(input, constant).expect("add is valid");
    let relu = builder.relu(sum).expect("relu is valid");
    let twice = builder.add(sum, sum).expect("shared add is valid");
    for output in [twice, relu, sum, relu, input] {
        builder.mark_output(output).expect("output is valid");
    }
    let graph = builder.build().expect("graph is valid");
    let encoded = encode_graph_v1(&graph).expect("graph is supported");
    let fixture = include_str!("../../../proofs/fixtures/strict-graph-v1.json");

    assert_eq!(encoded, fixture.trim_end_matches('\n'));
    assert_eq!(encode_graph_v1(&graph).expect("repeat encoding"), encoded);
    let decoded: Value = serde_json::from_str(&encoded).expect("encoding is JSON");
    assert_eq!(decoded[2][0][2], "features\"\\\n雪");
    assert_eq!(decoded[3], serde_json::json!([4, 3, 2, 3, 0]));
}

#[test]
fn scalar_shapes_and_input_appearance_order_are_preserved() {
    let mut builder = GraphBuilder::new();
    let first = builder
        .input("z", DType::F64, shape(&[]))
        .expect("first scalar input");
    let constant = builder
        .constant(TensorData::F64(vec![f64::from_bits(1)]), shape(&[]))
        .expect("scalar minimum-subnormal constant");
    let second = builder
        .input("a", DType::F64, shape(&[]))
        .expect("second scalar input");
    let sum = builder.add(first, constant).expect("scalar add");
    for output in [second, sum] {
        builder.mark_output(output).expect("scalar output");
    }
    let graph = builder.build().expect("scalar graph");

    assert_eq!(
        encode_graph_v1(&graph).expect("scalar encoding"),
        r#"["vtc.graph",1,[["input","f64","z",[]],["const","f64",[],["0000000000000001"]],["input","f64","a",[]],["add",0,1]],[2,3]]"#,
    );
}

#[test]
fn duplicate_input_names_are_rejected_even_when_unused() {
    let mut builder = GraphBuilder::new();
    let first = builder
        .input("x", DType::F64, shape(&[]))
        .expect("first input");
    let _second = builder
        .input("x", DType::F64, shape(&[]))
        .expect("second input");
    builder.mark_output(first).expect("output");

    assert!(matches!(
        encode_graph_v1(&builder.build().expect("structurally valid")),
        Err(GraphEncodeError::DuplicateInputName { name }) if name == "x"
    ));
}

#[test]
fn zero_dimensions_and_shape_overflow_are_rejected() {
    for dims in [&[0][..], &[usize::MAX, 2][..]] {
        let mut builder = GraphBuilder::new();
        let input = builder
            .input("x", DType::F64, shape(dims))
            .expect("input creation defers shape limits");
        builder.mark_output(input).expect("output");
        assert!(encode_graph_v1(&builder.build().expect("structurally valid")).is_err());
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn source_shape_sizes_use_unsigned_integers_without_kernel_allocation() {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input("large", DType::F64, shape(&[usize::MAX]))
        .expect("shape metadata does not allocate tensor elements");
    builder.mark_output(input).expect("output");

    assert_eq!(
        encode_graph_v1(&builder.build().expect("structurally valid")).expect("u64 size fits"),
        r#"["vtc.graph",1,[["input","f64","large",[18446744073709551615]]],[0]]"#,
    );
}

#[test]
fn nonfinite_constants_are_rejected() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut builder = GraphBuilder::new();
        let constant = builder
            .constant(TensorData::F64(vec![value]), shape(&[]))
            .expect("builder permits nonfinite constants");
        builder.mark_output(constant).expect("output");
        assert!(matches!(
            encode_graph_v1(&builder.build().expect("structurally valid")),
            Err(GraphEncodeError::NonFiniteConstant { index: 0, .. })
        ));
    }
}

#[test]
fn other_dtypes_and_operations_are_rejected() {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input("x", DType::F32, shape(&[]))
        .expect("f32 input");
    builder.mark_output(input).expect("output");
    assert!(matches!(
        encode_graph_v1(&builder.build().expect("valid f32 graph")),
        Err(GraphEncodeError::UnsupportedDtype {
            dtype: DType::F32,
            ..
        })
    ));

    let mut builder = GraphBuilder::new();
    let input = builder
        .input("x", DType::F64, shape(&[]))
        .expect("f64 input");
    let neg = builder.neg(input).expect("neg is a valid IR operation");
    builder.mark_output(neg).expect("output");
    assert!(matches!(
        encode_graph_v1(&builder.build().expect("valid neg graph")),
        Err(GraphEncodeError::UnsupportedOp { op: "neg", .. })
    ));
}

#[test]
fn illtyped_supported_operations_are_rejected() {
    let mut builder = GraphBuilder::new();
    let scalar = builder
        .input("scalar", DType::F64, shape(&[]))
        .expect("scalar input");
    let vector = builder
        .input("vector", DType::F64, shape(&[1]))
        .expect("vector input");
    let sum = builder
        .add(scalar, vector)
        .expect("builder checks only ids");
    builder.mark_output(sum).expect("output");
    assert!(matches!(
        encode_graph_v1(&builder.build().expect("structurally valid")),
        Err(GraphEncodeError::InvalidGraph(_))
    ));
}
