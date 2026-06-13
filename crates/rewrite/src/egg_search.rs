//! Untrusted egg search guarded by interpreter oracles.

#![allow(
    clippy::module_name_repetitions,
    reason = "step-10 public API names intentionally include Egg"
)]

use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use egg::{CostFunction, Extractor, Id, Pattern, RecExpr, Rewrite as EggRewrite, Runner};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_interp::{Tensor, TensorF64, eval, eval_f64};
use vtc_ir::{DType, Graph, Op};

use crate::egg_lang::{
    AtomTable, EggError, EggLang, graph_to_recexpr, recexpr_to_graph, recexprs_to_graph,
};
use crate::{RewriteMode, prune_and_rebuild};

/// egg rewrite rule type for the VTC egg language.
pub type EggRule = EggRewrite<EggLang, ()>;

/// Configuration for untrusted egg search and oracle validation.
#[derive(Debug, Clone)]
pub struct EggConfig {
    /// Rewrite mode controlling egg rules and validation gates.
    pub mode: RewriteMode,
    /// Maximum e-graph node count.
    pub node_limit: usize,
    /// Maximum equality-saturation iterations.
    pub iter_limit: usize,
    /// Maximum runner wall-clock time.
    pub time_limit: Duration,
    /// Number of random validation trials.
    pub validation_trials: usize,
    /// Base seed for deterministic validation inputs.
    pub seed: u64,
}

impl Default for EggConfig {
    fn default() -> Self {
        Self {
            mode: RewriteMode::Strict,
            node_limit: 10_000,
            iter_limit: 8,
            time_limit: Duration::from_millis(250),
            validation_trials: 64,
            seed: 0x517c_c1b7_5eed_0100,
        }
    }
}

/// Result of an egg optimization attempt.
#[derive(Debug, Clone)]
pub struct EggResult {
    /// Original graph or accepted optimized graph.
    pub graph: Graph,
    /// Whether the extracted candidate cleared the oracle gate.
    pub accepted: bool,
    /// Original node count.
    pub original_nodes: usize,
    /// Returned graph node count.
    pub result_nodes: usize,
    /// Number of validation trials requested.
    pub validation_trials: usize,
}

/// Builds an egg rewrite rule from pattern strings.
///
/// The resulting rule is untrusted until a candidate produced from it clears
/// the oracle gate.
///
/// # Errors
///
/// Returns [`EggError`] if either pattern cannot be parsed or egg rejects the
/// rewrite.
pub fn make_egg_rule(
    name: &'static str,
    searcher: &str,
    applier: &str,
) -> Result<EggRule, EggError> {
    let searcher = Pattern::<EggLang>::from_str(searcher)
        .map_err(|error| EggError::ConversionFailed(format!("{error}")))?;
    let applier = Pattern::<EggLang>::from_str(applier)
        .map_err(|error| EggError::ConversionFailed(format!("{error}")))?;
    EggRewrite::new(name, searcher, applier)
        .map_err(|error| EggError::ConversionFailed(error.clone()))
}

/// Returns the untrusted egg rule set enabled by `mode`.
///
/// Strict mode includes only bit-exact rules. Fast-math mode also includes
/// real-only associativity rules.
///
/// # Errors
///
/// Returns [`EggError`] if a built-in egg pattern cannot be constructed.
pub fn egg_rules_for_mode(mode: RewriteMode) -> Result<Vec<EggRule>, EggError> {
    let mut rules = vec![
        make_egg_rule("egg-neg-neg", "(neg (neg ?x))", "?x")?,
        make_egg_rule("egg-relu-relu", "(relu (relu ?x))", "(relu ?x)")?,
        make_egg_rule("egg-add-comm", "(add ?a ?b)", "(add ?b ?a)")?,
        make_egg_rule("egg-mul-comm", "(mul ?a ?b)", "(mul ?b ?a)")?,
    ];
    if matches!(mode, RewriteMode::FastMath) {
        rules.push(make_egg_rule(
            "egg-add-assoc-right",
            "(add (add ?a ?b) ?c)",
            "(add ?a (add ?b ?c))",
        )?);
        rules.push(make_egg_rule(
            "egg-add-assoc-left",
            "(add ?a (add ?b ?c))",
            "(add (add ?a ?b) ?c)",
        )?);
    }
    Ok(rules)
}

/// Optimizes a graph with untrusted egg search and oracle validation.
///
/// The oracle gate is the trust boundary: rational evaluation must match on
/// every validation trial, and strict mode additionally requires IEEE `f64`
/// bit equality. A failing or invalid candidate is rejected and the original
/// graph is returned with `accepted = false`.
///
/// # Errors
///
/// Returns [`EggError`] if the original graph cannot be converted or the
/// built-in egg rules cannot be constructed.
pub fn optimize_with_egg(graph: &Graph, cfg: &EggConfig) -> Result<EggResult, EggError> {
    let rules = egg_rules_for_mode(cfg.mode)?;
    optimize_with_egg_rules(graph, cfg, &rules)
}

/// Optimizes a graph with caller-supplied untrusted egg rules.
///
/// This is primarily useful for tests that inject intentionally unsound rules.
/// Candidates still must clear the same oracle gate as [`optimize_with_egg`].
///
/// # Errors
///
/// Returns [`EggError`] if the original graph cannot be converted.
pub fn optimize_with_egg_rules(
    graph: &Graph,
    cfg: &EggConfig,
    rules: &[EggRule],
) -> Result<EggResult, EggError> {
    graph.validate_structure()?;
    let original =
        prune_and_rebuild(graph).map_err(|error| EggError::Internal(error.to_string()))?;
    let original_nodes = original.num_nodes();
    let extracted = extract_outputs(&original, cfg, rules)?;
    let Ok(candidate) = back_convert_candidate(&extracted, &original) else {
        return Ok(rejected(original, original_nodes, cfg.validation_trials));
    };

    if candidate.num_nodes() > original_nodes || !validate_candidate(&original, &candidate, cfg) {
        return Ok(rejected(original, original_nodes, cfg.validation_trials));
    }

    let result_nodes = candidate.num_nodes();
    Ok(EggResult {
        graph: candidate,
        accepted: true,
        original_nodes,
        result_nodes,
        validation_trials: cfg.validation_trials,
    })
}

fn extract_outputs(
    graph: &Graph,
    cfg: &EggConfig,
    rules: &[EggRule],
) -> Result<Vec<(RecExpr<EggLang>, AtomTable)>, EggError> {
    graph
        .outputs()
        .iter()
        .map(|&output| {
            let (expr, atoms) = graph_to_recexpr(graph, output)?;
            let extracted = extract_one(&expr, cfg, rules);
            Ok((extracted, atoms))
        })
        .collect()
}

fn extract_one(expr: &RecExpr<EggLang>, cfg: &EggConfig, rules: &[EggRule]) -> RecExpr<EggLang> {
    let runner = Runner::<EggLang, ()>::default()
        .with_expr(expr)
        .with_node_limit(cfg.node_limit)
        .with_iter_limit(cfg.iter_limit)
        .with_time_limit(cfg.time_limit)
        .run(rules);
    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, RightAssocCost);
    let (_, best) = extractor.find_best(root);
    best
}

fn back_convert_candidate(
    extracted: &[(RecExpr<EggLang>, AtomTable)],
    original: &Graph,
) -> Result<Graph, EggError> {
    if extracted.len() == 1 {
        let (expr, atoms) = &extracted[0];
        return recexpr_to_graph(expr, atoms, original);
    }
    let outputs = extracted
        .iter()
        .map(|(expr, atoms)| (expr, atoms))
        .collect::<Vec<_>>();
    recexprs_to_graph(&outputs, original)
}

fn validate_candidate(original: &Graph, candidate: &Graph, cfg: &EggConfig) -> bool {
    (0..cfg.validation_trials).all(|trial| {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(trial as u64));
        let Some(exact_inputs) = random_exact_inputs(original, &mut rng) else {
            return false;
        };
        let Some(original_outputs) = eval(original, &exact_inputs).ok() else {
            return false;
        };
        let Some(candidate_outputs) = eval(candidate, &exact_inputs).ok() else {
            return false;
        };
        if original_outputs != candidate_outputs {
            return false;
        }
        if matches!(cfg.mode, RewriteMode::Strict) {
            let Some(f64_inputs) = random_f64_inputs(original, &mut rng) else {
                return false;
            };
            let Some(original_outputs) = eval_f64(original, &f64_inputs).ok() else {
                return false;
            };
            let Some(candidate_outputs) = eval_f64(candidate, &f64_inputs).ok() else {
                return false;
            };
            original_outputs.len() == candidate_outputs.len()
                && original_outputs
                    .iter()
                    .zip(candidate_outputs.iter())
                    .all(|(left, right)| left.bit_eq(right))
        } else {
            true
        }
    })
}

fn random_exact_inputs(graph: &Graph, rng: &mut StdRng) -> Option<HashMap<String, Tensor>> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).ok()?;
        let Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().ok()?;
        let tensor = match ty.dtype() {
            DType::F32 | DType::F64 => {
                let values = random_f64_values(numel, rng);
                Tensor::from_f64(ty.shape().clone(), &values).ok()?
            }
            DType::I32 | DType::I64 => {
                let values = random_i64_values(numel, rng);
                Tensor::from_i64(ty.shape().clone(), &values).ok()?
            }
            DType::Bool => return None,
        };
        inputs.insert(name.clone(), tensor);
    }
    Some(inputs)
}

fn random_f64_inputs(graph: &Graph, rng: &mut StdRng) -> Option<HashMap<String, TensorF64>> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).ok()?;
        let Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().ok()?;
        let values = match ty.dtype() {
            DType::F32 | DType::F64 => random_f64_values(numel, rng),
            DType::I32 | DType::I64 => random_i64_values(numel, rng)
                .into_iter()
                .map(clamped_f64)
                .collect(),
            DType::Bool => return None,
        };
        let tensor = TensorF64::from_f64(ty.shape().clone(), &values).ok()?;
        inputs.insert(name.clone(), tensor);
    }
    Some(inputs)
}

fn random_f64_values(numel: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..numel).map(|_| random_sensitive_f64(rng)).collect()
}

fn random_i64_values(numel: usize, rng: &mut StdRng) -> Vec<i64> {
    (0..numel).map(|_| rng.gen_range(-8..=8)).collect()
}

fn random_sensitive_f64(rng: &mut StdRng) -> f64 {
    match rng.gen_range(0..12) {
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

fn clamped_f64(value: i64) -> f64 {
    f64::from(clamp_i32(value))
}

fn clamp_i32(value: i64) -> i32 {
    match i32::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => i32::MIN,
        Err(_) => i32::MAX,
    }
}

fn rejected(graph: Graph, original_nodes: usize, validation_trials: usize) -> EggResult {
    EggResult {
        result_nodes: graph.num_nodes(),
        graph,
        accepted: false,
        original_nodes,
        validation_trials,
    }
}

#[derive(Debug, Clone, Copy)]
struct RightAssocCost;

impl CostFunction<EggLang> for RightAssocCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &EggLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            EggLang::Atom(_) => 1,
            EggLang::Neg([input]) | EggLang::Relu([input]) => 1 + costs(*input),
            EggLang::Matmul([left, right]) => 8 + costs(*left) + costs(*right),
            EggLang::Add([left, right]) => 1 + (2 * costs(*left)) + costs(*right),
            EggLang::Sub([left, right]) | EggLang::Mul([left, right]) => {
                1 + costs(*left) + costs(*right)
            }
        }
    }
}
