//! Integration tests for untrusted egg search behind oracle validation.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::{Tensor, TensorF64, eval, eval_f64};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};
use vtc_rewrite::{
    EggConfig, RewriteMode, graph_to_recexpr, make_egg_rule, optimize_with_egg,
    optimize_with_egg_rules, recexpr_to_graph,
};

const RANDOM_GRAPHS: usize = 32;
const F64_INPUT_SETS: usize = 4;

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output id is valid");
    builder.build().expect("graph is structurally valid")
}

#[test]
fn conversion_round_trip_preserves_semantics_for_algebraic_and_atom_ops() {
    for graph in [algebraic_graph(), atom_heavy_graph()] {
        let output = graph.outputs()[0];
        let (expr, atoms) = graph_to_recexpr(&graph, output).expect("conversion succeeds");
        let round_trip = recexpr_to_graph(&expr, &atoms, &graph).expect("back-conversion succeeds");
        round_trip
            .validate_structure()
            .expect("round-trip graph is structurally valid");
        let exact_inputs = exact_inputs_for_graph(&graph);
        assert_exact_eval_eq(&graph, &round_trip, &exact_inputs);
    }
}

#[test]
fn strict_egg_simplifies_and_preserves_oracles() {
    let graph = algebraic_graph();
    let cfg = EggConfig {
        mode: RewriteMode::Strict,
        seed: 0xd00,
        ..EggConfig::default()
    };
    let result = optimize_with_egg(&graph, &cfg).expect("egg optimization succeeds");
    let exact_inputs = exact_inputs_for_graph(&graph);
    let f64_inputs = f64_inputs_for_graph(&graph);

    assert!(result.accepted);
    assert!(result.result_nodes <= result.original_nodes);
    assert_exact_eval_eq(&graph, &result.graph, &exact_inputs);
    assert_f64_eval_bit_eq(&graph, &result.graph, &f64_inputs);
}

#[test]
fn validation_gate_rejects_unsound_egg_rule() {
    let graph = add_const_graph();
    let cfg = EggConfig {
        mode: RewriteMode::Strict,
        seed: 0xe00,
        validation_trials: 16,
        ..EggConfig::default()
    };
    let unsound =
        make_egg_rule("unsound-add-left", "(add ?a ?b)", "?a").expect("test rule pattern is valid");
    let result = optimize_with_egg_rules(&graph, &cfg, &[unsound])
        .expect("egg optimization returns fallback");

    assert!(!result.accepted);
    assert_eq!(result.graph.to_text(), graph.to_text());
    assert_eq!(result.original_nodes, graph.num_nodes());
    assert_eq!(result.result_nodes, graph.num_nodes());
}

#[test]
fn strict_excludes_and_fast_math_accepts_reassociation() {
    let graph = add_assoc_graph();
    let strict_cfg = EggConfig {
        mode: RewriteMode::Strict,
        seed: 0xf00,
        ..EggConfig::default()
    };
    let fast_cfg = EggConfig {
        mode: RewriteMode::FastMath,
        seed: 0xf00,
        ..EggConfig::default()
    };
    let inputs = assoc_sensitive_inputs();

    let strict = optimize_with_egg(&graph, &strict_cfg).expect("strict egg succeeds");
    assert_f64_eval_bit_eq(&graph, &strict.graph, &inputs);

    let fast = optimize_with_egg(&graph, &fast_cfg).expect("fast-math egg succeeds");
    assert!(fast.accepted);
    assert_exact_eval_eq(&graph, &fast.graph, &assoc_exact_inputs());
    assert!(
        !eval_f64(&graph, &inputs).expect("original evaluates")[0]
            .bit_eq(&eval_f64(&fast.graph, &inputs).expect("optimized evaluates")[0])
    );
}

#[test]
fn strict_random_graph_outputs_clear_rational_and_f64_oracles() {
    let cfg = GenConfig {
        dtype: DType::F64,
        seed: 0x1000,
        ..GenConfig::default()
    };
    let egg_cfg = EggConfig {
        mode: RewriteMode::Strict,
        seed: 0x1100,
        validation_trials: 32,
        ..EggConfig::default()
    };

    for iteration in 0..RANDOM_GRAPHS {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration as u64));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let result = optimize_with_egg(&graph, &egg_cfg).expect("egg optimization succeeds");
        let exact_inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        assert_exact_eval_eq(&graph, &result.graph, &exact_inputs);
        for inputs in f64_input_sets(&graph, &mut rng) {
            assert_f64_eval_bit_eq(&graph, &result.graph, &inputs);
        }
    }
}

#[test]
fn egg_search_is_deterministic_for_same_input_and_config() {
    let graph = algebraic_graph();
    let cfg = EggConfig {
        mode: RewriteMode::Strict,
        seed: 0x1200,
        ..EggConfig::default()
    };
    let left = optimize_with_egg(&graph, &cfg).expect("left run succeeds");
    let right = optimize_with_egg(&graph, &cfg).expect("right run succeeds");

    assert_eq!(left.graph.to_text(), right.graph.to_text());
    assert_eq!(left.accepted, right.accepted);
    assert_eq!(left.original_nodes, right.original_nodes);
    assert_eq!(left.result_nodes, right.result_nodes);
}

fn algebraic_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", DType::F64, shape(&[2]))
        .expect("input is valid");
    let y = builder
        .input("y", DType::F64, shape(&[2]))
        .expect("input is valid");
    let neg = builder.neg(x).expect("neg is valid");
    let neg_neg = builder.neg(neg).expect("neg is valid");
    let relu = builder.relu(y).expect("relu is valid");
    let relu_relu = builder.relu(relu).expect("relu is valid");
    let add = builder.add(neg_neg, relu_relu).expect("add is valid");
    finish(builder, add)
}

fn atom_heavy_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", DType::F64, shape(&[2, 2]))
        .expect("input is valid");
    let reshaped = builder.reshape(x, shape(&[4])).expect("reshape is valid");
    let sum = builder.sum(reshaped, vec![0], false).expect("sum is valid");
    let neg = builder.neg(sum).expect("neg is valid");
    let neg_neg = builder.neg(neg).expect("neg is valid");
    finish(builder, neg_neg)
}

fn add_const_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", DType::F64, shape(&[1]))
        .expect("input is valid");
    let one = builder
        .constant(TensorData::F64(vec![1.0]), shape(&[1]))
        .expect("const is valid");
    let add = builder.add(x, one).expect("add is valid");
    finish(builder, add)
}

fn add_assoc_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let a = builder
        .input("a", DType::F64, shape(&[1]))
        .expect("input is valid");
    let b = builder
        .input("b", DType::F64, shape(&[1]))
        .expect("input is valid");
    let c = builder
        .input("c", DType::F64, shape(&[1]))
        .expect("input is valid");
    let left = builder.add(a, b).expect("add is valid");
    let root = builder.add(left, c).expect("add is valid");
    finish(builder, root)
}

fn exact_inputs_for_graph(graph: &Graph) -> HashMap<String, Tensor> {
    let mut rng = StdRng::seed_from_u64(0x1300);
    random_inputs(graph, 8, &mut rng)
}

fn f64_inputs_for_graph(graph: &Graph) -> HashMap<String, TensorF64> {
    let mut rng = StdRng::seed_from_u64(0x1400);
    random_f64_inputs(graph, &mut rng)
}

fn assoc_exact_inputs() -> HashMap<String, Tensor> {
    HashMap::from([
        (
            "a".to_owned(),
            Tensor::from_f64(shape(&[1]), &[1.0e16]).expect("input is valid"),
        ),
        (
            "b".to_owned(),
            Tensor::from_f64(shape(&[1]), &[-1.0e16]).expect("input is valid"),
        ),
        (
            "c".to_owned(),
            Tensor::from_f64(shape(&[1]), &[1.0]).expect("input is valid"),
        ),
    ])
}

fn assoc_sensitive_inputs() -> HashMap<String, TensorF64> {
    HashMap::from([
        (
            "a".to_owned(),
            TensorF64::from_f64(shape(&[1]), &[1.0e16]).expect("input is valid"),
        ),
        (
            "b".to_owned(),
            TensorF64::from_f64(shape(&[1]), &[-1.0e16]).expect("input is valid"),
        ),
        (
            "c".to_owned(),
            TensorF64::from_f64(shape(&[1]), &[1.0]).expect("input is valid"),
        ),
    ])
}

fn assert_exact_eval_eq(original: &Graph, optimized: &Graph, inputs: &HashMap<String, Tensor>) {
    assert_eq!(
        eval(original, inputs).expect("original evaluates"),
        eval(optimized, inputs).expect("optimized evaluates"),
    );
}

fn assert_f64_eval_bit_eq(
    original: &Graph,
    optimized: &Graph,
    inputs: &HashMap<String, TensorF64>,
) {
    let original_outputs = eval_f64(original, inputs).expect("original evaluates");
    let optimized_outputs = eval_f64(optimized, inputs).expect("optimized evaluates");
    assert_eq!(original_outputs.len(), optimized_outputs.len());
    for (left, right) in original_outputs.iter().zip(optimized_outputs.iter()) {
        assert!(left.bit_eq(right));
    }
}

fn f64_input_sets(graph: &Graph, rng: &mut StdRng) -> Vec<HashMap<String, TensorF64>> {
    (0..F64_INPUT_SETS)
        .map(|_| random_f64_inputs(graph, rng))
        .collect()
}

fn random_f64_inputs(graph: &Graph, rng: &mut StdRng) -> HashMap<String, TensorF64> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).expect("input id is valid");
        let vtc_ir::Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("shape is valid");
        let values = (0..numel)
            .map(|_| random_sensitive_f64(rng))
            .collect::<Vec<_>>();
        let tensor = TensorF64::from_f64(ty.shape().clone(), &values).expect("input is valid");
        inputs.insert(name.clone(), tensor);
    }
    inputs
}

fn random_sensitive_f64(rng: &mut StdRng) -> f64 {
    match rng.gen_range(0..10) {
        0 => -0.0,
        1 => 0.0,
        _ => {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            let mantissa: i32 = rng.gen_range(1..=16);
            let exponent: i32 = rng.gen_range(-4..=4);
            sign * f64::from(mantissa) * 2.0f64.powi(exponent)
        }
    }
}
