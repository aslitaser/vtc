//! Integration tests for the rewrite fixpoint driver.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::{Tensor, TensorF64, eval, eval_f64};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorData};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};
use vtc_rewrite::{
    DriverConfig, Law, NegNegElim, ReluIdempotentElim, ReshapeReshapeFuse, Rewrite, RewriteMode,
    RuleSet, replace_all_uses, run,
};

const RANDOM_RUNS: usize = 64;
const F64_INPUT_SETS: usize = 4;

fn default_ruleset() -> RuleSet {
    let mut rules = RuleSet::new();
    rules.add(Box::new(NegNegElim));
    rules.add(Box::new(ReluIdempotentElim));
    rules.add(Box::new(ReshapeReshapeFuse));
    rules
}

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn simplifiable_graph(dtype: DType) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", dtype, shape(&[4]))
        .expect("input is valid");
    let neg = builder.neg(x).expect("neg is valid");
    let neg_neg = builder.neg(neg).expect("neg is valid");
    let relu = builder.relu(neg_neg).expect("relu is valid");
    let relu_relu = builder.relu(relu).expect("relu is valid");
    let wide = builder
        .reshape(relu_relu, shape(&[2, 2]))
        .expect("reshape is valid");
    let flat = builder
        .reshape(wide, shape(&[4]))
        .expect("reshape is valid");
    builder.mark_output(flat).expect("output is valid");
    builder.build().expect("graph is valid")
}

#[test]
fn driver_reduces_simplifiable_graph_to_fixpoint() {
    let graph = simplifiable_graph(DType::F64);
    let rules = default_ruleset();
    let result = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");

    assert!(result.reached_fixpoint);
    assert!(result.steps() > 0);
    assert_eq!(
        rule_names(&result),
        [
            "neg-neg-elim",
            "relu-idempotent-elim",
            "reshape-reshape-fuse"
        ]
    );
    assert!(result.graph.num_nodes() < graph.num_nodes());
}

#[test]
fn driver_output_is_idempotent() {
    let graph = simplifiable_graph(DType::F64);
    let rules = default_ruleset();
    let first = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");
    let second = run(&first.graph, &rules, &DriverConfig::default()).expect("driver succeeds");

    assert_eq!(second.steps(), 0);
    assert!(second.reached_fixpoint);
    assert_eq!(first.graph.to_text(), second.graph.to_text());
}

#[test]
fn driver_is_deterministic() {
    let graph = simplifiable_graph(DType::F64);
    let rules = default_ruleset();
    let left = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");
    let right = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");

    assert_eq!(left.graph.to_text(), right.graph.to_text());
    assert_eq!(rule_names(&left), rule_names(&right));
    assert_eq!(left.steps(), right.steps());
    assert_eq!(left.reached_fixpoint, right.reached_fixpoint);
}

#[test]
fn driver_preserves_rational_semantics_on_random_graphs() {
    let cfg = GenConfig {
        seed: 0xb00,
        ..GenConfig::default()
    };
    let rules = default_ruleset();
    for iteration in 0..RANDOM_RUNS {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration as u64));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        let result = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");

        assert_exact_eval_eq(&graph, &result.graph, &inputs);
    }
}

#[test]
fn strict_driver_preserves_f64_bits_on_random_graphs() {
    let cfg = GenConfig {
        dtype: DType::F64,
        seed: 0xc00,
        ..GenConfig::default()
    };
    let rules = default_ruleset();
    for iteration in 0..RANDOM_RUNS {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration as u64));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let result = run(&graph, &rules, &DriverConfig::default()).expect("driver succeeds");
        for inputs in f64_input_sets(&graph, &mut rng) {
            assert_f64_eval_bit_eq(&graph, &result.graph, &inputs);
        }
    }
}

#[test]
fn driver_reports_cap_without_claiming_fixpoint() {
    let graph = simplifiable_graph(DType::F64);
    let rules = default_ruleset();
    let cfg = DriverConfig {
        mode: RewriteMode::Strict,
        max_steps: 1,
    };
    let result = run(&graph, &rules, &cfg).expect("driver succeeds");
    let inputs = HashMap::from([(
        "x".to_owned(),
        Tensor::from_f64(shape(&[4]), &[-0.0, 1.0, -2.0, 3.0]).expect("input is valid"),
    )]);

    assert!(!result.reached_fixpoint);
    assert_eq!(result.steps(), 1);
    result
        .graph
        .validate_structure()
        .expect("partial graph is valid");
    assert_exact_eval_eq(&graph, &result.graph, &inputs);
}

#[test]
fn mode_gating_keeps_real_only_rewrites_out_of_strict_mode() {
    let mut rules = default_ruleset();
    rules.add(Box::new(AddZeroElim));
    let (graph, exact_inputs, f64_inputs) = add_zero_graph_and_inputs();

    let strict = run(
        &graph,
        &rules,
        &DriverConfig {
            mode: RewriteMode::Strict,
            max_steps: 10,
        },
    )
    .expect("strict run succeeds");
    assert!(strict.reached_fixpoint);
    assert_eq!(strict.steps(), 0);
    assert!(strict.graph.to_text().contains("add"));
    assert_exact_eval_eq(&graph, &strict.graph, &exact_inputs);
    assert_f64_eval_bit_eq(&graph, &strict.graph, &f64_inputs);

    let fast_math = run(
        &graph,
        &rules,
        &DriverConfig {
            mode: RewriteMode::FastMath,
            max_steps: 10,
        },
    )
    .expect("fast-math run succeeds");
    assert!(fast_math.reached_fixpoint);
    assert_eq!(fast_math.steps(), 1);
    assert!(!fast_math.graph.to_text().contains("add"));
    assert_exact_eval_eq(&graph, &fast_math.graph, &exact_inputs);
    assert!(
        !eval_f64(&graph, &f64_inputs).expect("original evaluates")[0]
            .bit_eq(&eval_f64(&fast_math.graph, &f64_inputs).expect("optimized evaluates")[0])
    );
}

struct AddZeroElim;

impl Rewrite for AddZeroElim {
    fn name(&self) -> &'static str {
        "add-zero-elim"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::AddZeroIdentity]
    }

    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph> {
        let Op::Add(left, right) = graph.node(node).ok()?.op() else {
            return None;
        };
        if zero_const(graph, *right) {
            return replace_all_uses(graph, node, *left).ok();
        }
        if zero_const(graph, *left) {
            return replace_all_uses(graph, node, *right).ok();
        }
        None
    }
}

fn add_zero_graph_and_inputs() -> (Graph, HashMap<String, Tensor>, HashMap<String, TensorF64>) {
    let graph = add_zero_graph();
    let exact_inputs = HashMap::from([(
        "x".to_owned(),
        Tensor::from_f64(shape(&[4]), &[-0.0, 1.0, 0.0, -2.0]).expect("input is valid"),
    )]);
    let f64_inputs = HashMap::from([(
        "x".to_owned(),
        TensorF64::from_f64(shape(&[4]), &[-0.0, 1.0, 0.0, -2.0]).expect("input is valid"),
    )]);
    (graph, exact_inputs, f64_inputs)
}

fn add_zero_graph() -> Graph {
    let graph_shape = shape(&[4]);
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", DType::F64, graph_shape.clone())
        .expect("input is valid");
    let zero = builder
        .constant(TensorData::F64(vec![0.0; 4]), graph_shape)
        .expect("zero const is valid");
    let add = builder.add(x, zero).expect("add is valid");
    builder.mark_output(add).expect("output is valid");
    builder.build().expect("graph is valid")
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
        let node = graph.node(input_id).expect("generated input id is valid");
        let Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("generated shape numel is valid");
        let values = (0..numel)
            .map(|_| random_sensitive_f64(rng))
            .collect::<Vec<_>>();
        let tensor =
            TensorF64::from_f64(ty.shape().clone(), &values).expect("generated tensor is valid");
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

fn rule_names(result: &vtc_rewrite::RunResult) -> Vec<&'static str> {
    result.applied.iter().map(|applied| applied.rule).collect()
}

fn zero_const(graph: &Graph, node: NodeId) -> bool {
    let Ok(node) = graph.node(node) else {
        return false;
    };
    let Op::Const { data, .. } = node.op() else {
        return false;
    };
    match data {
        TensorData::F32(values) => values.iter().all(|&value| {
            value.to_bits() == 0.0f32.to_bits() || value.to_bits() == (-0.0f32).to_bits()
        }),
        TensorData::F64(values) => values.iter().all(|&value| {
            value.to_bits() == 0.0f64.to_bits() || value.to_bits() == (-0.0f64).to_bits()
        }),
        TensorData::I32(values) => values.iter().all(|&value| value == 0),
        TensorData::I64(values) => values.iter().all(|&value| value == 0),
        TensorData::Bool(_) => false,
    }
}
