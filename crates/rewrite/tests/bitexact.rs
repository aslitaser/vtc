//! IEEE `f64` bit-exactness tests for rewrite safety tags.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::{TensorF64, eval_f64};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorData};
use vtc_rewrite::r#gen::{GenConfig, Pattern, graph_with_pattern, random_graph};
use vtc_rewrite::{
    Law, NegNegElim, NumericSafety, ReluIdempotentElim, ReshapeReshapeFuse, Rewrite,
    replace_all_uses,
};

const PATTERN_ITERATIONS: usize = 200;
const INPUT_ASSIGNMENTS: usize = 4;

#[derive(Debug)]
struct BitMismatch {
    rule: String,
    seed: u64,
    iteration: usize,
    node: NodeId,
    original_text: String,
    rewritten_text: String,
    output_index: usize,
}

struct CheckContext<'a> {
    rule: &'a dyn Rewrite,
    seed: u64,
    iteration: usize,
    node: NodeId,
}

fn check_bit_exact(
    rule: &dyn Rewrite,
    cfg: &GenConfig,
    pattern: Option<Pattern>,
    iterations: usize,
) -> Result<(), BitMismatch> {
    if rule.safety() != NumericSafety::BitExact {
        return Ok(());
    }

    let cfg = GenConfig {
        dtype: DType::F64,
        ..cfg.clone()
    };
    for iteration in 0..iterations {
        let seed = cfg.seed.wrapping_add(iteration as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let (graph, _) = match pattern {
            Some(pattern) => graph_with_pattern(&cfg, &mut rng, pattern),
            None => random_graph(&cfg, &mut rng),
        };
        let inputs = (0..INPUT_ASSIGNMENTS)
            .map(|_| random_f64_inputs(&graph, &mut rng))
            .collect::<Vec<_>>();
        check_graph_bit_exact(rule, seed, iteration, &graph, &inputs)?;
    }

    Ok(())
}

fn check_graph_bit_exact(
    rule: &dyn Rewrite,
    seed: u64,
    iteration: usize,
    graph: &Graph,
    input_sets: &[HashMap<String, TensorF64>],
) -> Result<(), BitMismatch> {
    if rule.safety() != NumericSafety::BitExact {
        return Ok(());
    }

    for node in graph.topo_order().expect("test graph is a DAG") {
        let Some(rewritten) = rule.try_at(graph, node) else {
            continue;
        };
        let ctx = CheckContext {
            rule,
            seed,
            iteration,
            node,
        };
        for inputs in input_sets {
            let original_outputs =
                eval_f64(graph, inputs).map_err(|_| bit_mismatch(&ctx, graph, &rewritten, 0))?;
            let rewritten_outputs = eval_f64(&rewritten, inputs)
                .map_err(|_| bit_mismatch(&ctx, graph, &rewritten, 0))?;
            if original_outputs.len() != rewritten_outputs.len() {
                return Err(bit_mismatch(&ctx, graph, &rewritten, usize::MAX));
            }
            for (output_index, (left, right)) in original_outputs
                .iter()
                .zip(rewritten_outputs.iter())
                .enumerate()
            {
                if !left.bit_eq(right) {
                    return Err(bit_mismatch(&ctx, graph, &rewritten, output_index));
                }
            }
        }
    }

    Ok(())
}

fn bit_mismatch(
    ctx: &CheckContext<'_>,
    original: &Graph,
    rewritten: &Graph,
    output_index: usize,
) -> BitMismatch {
    BitMismatch {
        rule: ctx.rule.name().to_owned(),
        seed: ctx.seed,
        iteration: ctx.iteration,
        node: ctx.node,
        original_text: original.to_text(),
        rewritten_text: rewritten.to_text(),
        output_index,
    }
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

#[test]
fn neg_neg_is_f64_bit_exact() {
    let cfg = GenConfig {
        seed: 0x700,
        ..GenConfig::default()
    };
    check_bit_exact(&NegNegElim, &cfg, Some(Pattern::NegNeg), PATTERN_ITERATIONS)
        .expect("neg-neg rewrite is f64 bit-exact");
}

#[test]
fn relu_idempotent_is_f64_bit_exact() {
    let cfg = GenConfig {
        seed: 0x800,
        ..GenConfig::default()
    };
    check_bit_exact(
        &ReluIdempotentElim,
        &cfg,
        Some(Pattern::ReluRelu),
        PATTERN_ITERATIONS,
    )
    .expect("relu-idempotent rewrite is f64 bit-exact");
}

#[test]
fn reshape_reshape_is_f64_bit_exact() {
    let cfg = GenConfig {
        seed: 0x900,
        ..GenConfig::default()
    };
    check_bit_exact(
        &ReshapeReshapeFuse,
        &cfg,
        Some(Pattern::ReshapeReshape),
        PATTERN_ITERATIONS,
    )
    .expect("reshape-reshape rewrite is f64 bit-exact");
}

struct BogusAddZeroElim;

impl Rewrite for BogusAddZeroElim {
    fn name(&self) -> &'static str {
        "bogus-add-zero-elim"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::StructuralOnly]
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

#[test]
fn bogus_add_zero_elim_is_flagged_on_signed_zero() {
    let (graph, add, inputs) = add_zero_graph_with_negative_zero_input();
    let original = eval_f64(&graph, &inputs).expect("original graph evaluates");
    let rewritten = BogusAddZeroElim
        .try_at(&graph, add)
        .expect("bogus rewrite fires");
    let rewritten_outputs = eval_f64(&rewritten, &inputs).expect("rewritten graph evaluates");
    assert!(!original[0].bit_eq(&rewritten_outputs[0]));

    let mismatch = check_graph_bit_exact(&BogusAddZeroElim, 0xa00, 0, &graph, &[inputs])
        .expect_err("bogus add-zero rewrite must be detected");
    assert_eq!(mismatch.rule, "bogus-add-zero-elim");
    assert_eq!(mismatch.seed, 0xa00);
    assert_eq!(mismatch.iteration, 0);
    assert_eq!(mismatch.output_index, 0);
    assert!(mismatch.original_text.contains("add"));
    assert!(mismatch.rewritten_text.contains("outputs: %0"));
    assert_eq!(mismatch.node, add);
}

fn add_zero_graph_with_negative_zero_input() -> (Graph, NodeId, HashMap<String, TensorF64>) {
    let shape = Shape::new(vec![Dim::new(4)]);
    let mut builder = GraphBuilder::new();
    let x = builder
        .input("x", DType::F64, shape.clone())
        .expect("input is valid");
    let zero = builder
        .constant(TensorData::F64(vec![0.0; 4]), shape.clone())
        .expect("zero const is valid");
    let add = builder.add(x, zero).expect("add operands are valid");
    builder.mark_output(add).expect("output id is valid");
    let graph = builder.build().expect("graph is valid");
    let input = TensorF64::from_f64(shape, &[-0.0, 1.0, 0.0, -2.0]).expect("input is valid");

    (graph, add, HashMap::from([("x".to_owned(), input)]))
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
