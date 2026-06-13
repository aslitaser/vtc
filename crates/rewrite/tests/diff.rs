//! Differential tests for rewrite semantic preservation.

use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::eval;
use vtc_ir::{Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorData, infer_types};
use vtc_rewrite::r#gen::{GenConfig, Pattern, graph_with_pattern, random_graph, random_inputs};
use vtc_rewrite::{
    Law, NegNegElim, ReluIdempotentElim, ReshapeReshapeFuse, Rewrite, RewriteMode, RuleSet,
    prune_and_rebuild, redefine_node,
};

const PATTERN_ITERATIONS: usize = 200;
const RANDOM_ITERATIONS: usize = 200;
const INPUT_ASSIGNMENTS: usize = 3;

#[derive(Debug)]
struct Mismatch {
    rule: String,
    seed: u64,
    iteration: usize,
    node: NodeId,
    original: String,
    rewritten: String,
    output_index: usize,
}

struct CheckContext<'a> {
    rule: &'a dyn Rewrite,
    seed: u64,
    iteration: usize,
    node: NodeId,
    cfg: &'a GenConfig,
}

fn check_rule_preserves_semantics(
    rule: &dyn Rewrite,
    cfg: &GenConfig,
    pattern: Option<Pattern>,
    iterations: usize,
) -> Result<(), Mismatch> {
    // This harness checks exact rational equivalence through the reference
    // interpreter. It catches wrong rewrites and graph-surgery bugs, but it is
    // not evidence that a rewrite's BitExact/RealOnly tag is correct for IEEE.
    for iteration in 0..iterations {
        let seed = cfg.seed.wrapping_add(iteration as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let (graph, _) = match pattern {
            Some(pattern) => graph_with_pattern(cfg, &mut rng, pattern),
            None => random_graph(cfg, &mut rng),
        };

        check_graph_preserves_semantics(rule, cfg, seed, iteration, &graph, &mut rng)?;
    }

    Ok(())
}

fn check_graph_preserves_semantics(
    rule: &dyn Rewrite,
    cfg: &GenConfig,
    seed: u64,
    iteration: usize,
    graph: &Graph,
    rng: &mut StdRng,
) -> Result<(), Mismatch> {
    for node in graph.topo_order().expect("test graph is a DAG") {
        let Some(rewritten) = rule.try_at(graph, node) else {
            continue;
        };
        let ctx = CheckContext {
            rule,
            seed,
            iteration,
            node,
            cfg,
        };
        infer_types(&rewritten).map_err(|_| mismatch(&ctx, graph, &rewritten, 0))?;
        check_eval_equal(&ctx, graph, &rewritten, rng)?;

        let pruned =
            prune_and_rebuild(&rewritten).map_err(|_| mismatch(&ctx, graph, &rewritten, 0))?;
        infer_types(&pruned).map_err(|_| mismatch(&ctx, graph, &pruned, 0))?;
        check_eval_equal(&ctx, graph, &pruned, rng)?;
    }

    Ok(())
}

fn check_eval_equal(
    ctx: &CheckContext<'_>,
    original: &Graph,
    rewritten: &Graph,
    rng: &mut StdRng,
) -> Result<(), Mismatch> {
    for _ in 0..INPUT_ASSIGNMENTS {
        let inputs = random_inputs(original, ctx.cfg.value_range, rng);
        let original_outputs =
            eval(original, &inputs).map_err(|_| mismatch(ctx, original, rewritten, 0))?;
        let rewritten_outputs =
            eval(rewritten, &inputs).map_err(|_| mismatch(ctx, original, rewritten, 0))?;
        if original_outputs.len() != rewritten_outputs.len() {
            return Err(mismatch(ctx, original, rewritten, usize::MAX));
        }
        for (output_index, (left, right)) in original_outputs
            .iter()
            .zip(rewritten_outputs.iter())
            .enumerate()
        {
            if left != right {
                return Err(mismatch(ctx, original, rewritten, output_index));
            }
        }
    }

    Ok(())
}

fn mismatch(
    ctx: &CheckContext<'_>,
    original: &Graph,
    rewritten: &Graph,
    output_index: usize,
) -> Mismatch {
    Mismatch {
        rule: ctx.rule.name().to_owned(),
        seed: ctx.seed,
        iteration: ctx.iteration,
        node: ctx.node,
        original: original.to_text(),
        rewritten: rewritten.to_text(),
        output_index,
    }
}

#[test]
fn neg_neg_rewrite_preserves_rational_semantics() {
    let cfg = GenConfig {
        seed: 0x100,
        ..GenConfig::default()
    };
    check_rule_preserves_semantics(&NegNegElim, &cfg, Some(Pattern::NegNeg), PATTERN_ITERATIONS)
        .expect("neg-neg rewrite preserves exact rational semantics");
}

#[test]
fn relu_idempotent_rewrite_preserves_rational_semantics() {
    let cfg = GenConfig {
        seed: 0x200,
        ..GenConfig::default()
    };
    check_rule_preserves_semantics(
        &ReluIdempotentElim,
        &cfg,
        Some(Pattern::ReluRelu),
        PATTERN_ITERATIONS,
    )
    .expect("relu-idempotent rewrite preserves exact rational semantics");
}

#[test]
fn reshape_reshape_rewrite_preserves_rational_semantics() {
    let cfg = GenConfig {
        seed: 0x300,
        ..GenConfig::default()
    };
    check_rule_preserves_semantics(
        &ReshapeReshapeFuse,
        &cfg,
        Some(Pattern::ReshapeReshape),
        PATTERN_ITERATIONS,
    )
    .expect("reshape-reshape rewrite preserves exact rational semantics");
}

#[test]
fn all_strict_rules_preserve_rational_semantics_on_random_graphs() {
    let cfg = GenConfig {
        seed: 0x400,
        ..GenConfig::default()
    };
    let mut rules = RuleSet::new();
    rules.add(Box::new(NegNegElim));
    rules.add(Box::new(ReluIdempotentElim));
    rules.add(Box::new(ReshapeReshapeFuse));

    for rule in rules.enabled(RewriteMode::Strict) {
        check_rule_preserves_semantics(rule, &cfg, None, RANDOM_ITERATIONS)
            .expect("strict rewrite preserves exact rational semantics");
    }
}

#[test]
fn generator_outputs_well_typed_evaluable_graphs() {
    let cfg = GenConfig {
        seed: 0x500,
        ..GenConfig::default()
    };

    for iteration in 0..100 {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration));
        let (graph, _) = random_graph(&cfg, &mut rng);
        infer_types(&graph).expect("generated graph is well typed");
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        eval(&graph, &inputs).expect("generated graph evaluates");
    }
}

struct BogusAddToMul;

impl Rewrite for BogusAddToMul {
    fn name(&self) -> &'static str {
        "bogus-add-to-mul"
    }

    fn laws(&self) -> &'static [Law] {
        &[Law::StructuralOnly]
    }

    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph> {
        let Op::Add(left, right) = graph.node(node).ok()?.op() else {
            return None;
        };
        redefine_node(graph, node, Op::Mul(*left, *right)).ok()
    }
}

#[test]
fn negative_control_flags_bogus_add_to_mul() {
    let cfg = GenConfig {
        seed: 0x600,
        size: 2,
        ..GenConfig::default()
    };
    let graph = add_graph();
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let mismatch =
        check_graph_preserves_semantics(&BogusAddToMul, &cfg, cfg.seed, 0, &graph, &mut rng)
            .expect_err("bogus rewrite must be detected");

    assert_eq!(mismatch.rule, "bogus-add-to-mul");
    assert_eq!(mismatch.iteration, 0);
    assert_eq!(mismatch.output_index, 0);
    assert!(mismatch.original.contains("add"));
    assert!(mismatch.rewritten.contains("mul"));
    assert_eq!(mismatch.seed, cfg.seed);
    assert!(mismatch.node.index() < 10);
}

fn add_graph() -> Graph {
    let shape = Shape::new(vec![Dim::new(1)]);
    let mut builder = GraphBuilder::new();
    let left = builder
        .constant(TensorData::I64(vec![2]), shape.clone())
        .expect("valid left const");
    let right = builder
        .constant(TensorData::I64(vec![3]), shape)
        .expect("valid right const");
    let output = builder.add(left, right).expect("valid add");
    builder.mark_output(output).expect("valid output");
    builder.build().expect("valid graph")
}
