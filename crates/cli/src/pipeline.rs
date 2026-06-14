//! End-to-end CLI pipeline composition.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;
use vtc_codegen::{
    BenchError, CodegenError, MeasuredCost, compile_and_run, emit_c, has_c_compiler,
};
use vtc_interp::{EvalError, Tensor, TensorF64, eval, eval_f64};
use vtc_ir::{Graph, IrError, Op, TypeError, infer_types};
use vtc_loopir::{Kernel, LoopError, LowerError, eval_loops, eval_loops_f64, lower};
use vtc_rewrite::{
    DriverConfig, EggConfig, EggError, NegNegElim, ReluIdempotentElim, ReshapeReshapeFuse,
    RewriteMode, RuleSet, SurgeryError, run,
};
use vtc_schedule::{
    CostModel, LegalityError, Mode, MoveDesc, StaticCost, TuneConfig, TuneError, autotune,
};

use crate::examples::build_example;

/// Schedule cost model selected by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostKind {
    /// Use the deterministic static heuristic.
    Static,
    /// Use measured C runtime when a compiler is available.
    Measured,
}

impl fmt::Display for CostKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static => f.write_str("static"),
            Self::Measured => f.write_str("measured"),
        }
    }
}

/// Options for an end-to-end pipeline run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "RunOpts mirrors the explicit boolean CLI flags from the capstone request"
)]
pub struct RunOpts {
    /// Built-in example name.
    pub example: String,
    /// Shape parameters such as `M=8`.
    pub shapes: HashMap<String, usize>,
    /// Rewrite and schedule legality mode.
    pub mode: Mode,
    /// Whether to run oracle-gated egg search after the rewrite driver.
    pub egg: bool,
    /// Schedule cost model.
    pub cost: CostKind,
    /// Kernel repetitions used by measured cost.
    pub reps: usize,
    /// Seed for deterministic generated inputs.
    pub seed: u64,
    /// Whether to run end-to-end self-checks.
    pub check: bool,
    /// Optional path to write emitted C.
    pub emit: Option<PathBuf>,
    /// Skip C compilation and execution.
    pub no_compile: bool,
    /// Include graph and kernel text in the report.
    pub verbose: bool,
}

/// Structured report from an end-to-end pipeline run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "RunReport records independent stage flags for command and test consumers"
)]
pub struct RunReport {
    /// Built-in example name.
    pub example: String,
    /// Input names in graph declaration order.
    pub input_names: Vec<String>,
    /// Mode used throughout the pipeline.
    pub mode: Mode,
    /// Whether egg was requested.
    pub egg_requested: bool,
    /// Cost model requested by the user.
    pub requested_cost: CostKind,
    /// Cost model actually used.
    pub used_cost: CostKind,
    /// Notes emitted during the run.
    pub notes: Vec<String>,
    /// Initial graph node count.
    pub nodes_initial: usize,
    /// Graph node count after rewrite driver.
    pub nodes_after_driver: usize,
    /// Graph node count after optional egg search.
    pub nodes_after_egg: usize,
    /// Names of rewrites applied by the driver.
    pub rewrites: Vec<String>,
    /// Whether the driver reached a true fixpoint.
    pub driver_fixpoint: bool,
    /// Whether egg accepted and the pipeline adopted a smaller candidate.
    pub egg_adopted: bool,
    /// Schedule moves chosen by the autotuner.
    pub schedule_moves: Vec<String>,
    /// Cost before schedule autotuning.
    pub cost_before: u64,
    /// Cost after schedule autotuning.
    pub cost_after: u64,
    /// Number of schedule candidates checked.
    pub candidates_evaluated: usize,
    /// Whether emitted C was written to disk.
    pub emitted_path: Option<PathBuf>,
    /// Emitted C source.
    pub emitted_c: String,
    /// Whether the C backend compiled and ran.
    pub compiled: bool,
    /// Truncated output tensors from compiled C, if compilation ran.
    pub output_preview: Vec<String>,
    /// Self-check status.
    pub check: CheckStatus,
    /// Verbose graph and kernel snapshots.
    pub verbose: Vec<String>,
}

impl fmt::Display for RunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "example: {}", self.example)?;
        writeln!(f, "inputs: {}", self.input_names.join(", "))?;
        writeln!(f, "mode: {}", mode_name(self.mode))?;
        writeln!(
            f,
            "graph: nodes {} -> {} after driver -> {} after egg",
            self.nodes_initial, self.nodes_after_driver, self.nodes_after_egg
        )?;
        writeln!(
            f,
            "rewrites: {} (fixpoint: {})",
            list_or_none(&self.rewrites),
            self.driver_fixpoint
        )?;
        writeln!(
            f,
            "egg: requested={}, adopted={}",
            self.egg_requested, self.egg_adopted
        )?;
        writeln!(
            f,
            "schedule: {} cost {} -> {} ({} candidates)",
            list_or_none(&self.schedule_moves),
            self.cost_before,
            self.cost_after,
            self.candidates_evaluated
        )?;
        writeln!(
            f,
            "cost model: requested={}, used={}",
            self.requested_cost, self.used_cost
        )?;
        if let Some(path) = &self.emitted_path {
            writeln!(f, "emit: {}", path.display())?;
        } else {
            writeln!(f, "emit: not written")?;
        }
        writeln!(f, "compiled: {}", self.compiled)?;
        if !self.output_preview.is_empty() {
            writeln!(f, "outputs: {}", self.output_preview.join("; "))?;
        }
        writeln!(f, "check: {}", self.check)?;
        for note in &self.notes {
            writeln!(f, "note: {note}")?;
        }
        for item in &self.verbose {
            writeln!(f, "\n{item}")?;
        }
        Ok(())
    }
}

/// Self-check status for a pipeline run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    /// Checks were not requested.
    NotRequested,
    /// Checks passed with these checks performed.
    Passed(Vec<String>),
    /// Checks failed with an explanatory message.
    Failed(String),
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRequested => f.write_str("not requested"),
            Self::Passed(checks) => write!(f, "self-check passed ({})", checks.join(", ")),
            Self::Failed(message) => write!(f, "self-check failed: {message}"),
        }
    }
}

/// Coarse run status for command consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Pipeline completed successfully.
    Success,
}

/// Errors produced by the CLI pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Unknown built-in example.
    #[error("unknown example {0:?}")]
    UnknownExample(String),
    /// Invalid shape parameter.
    #[error("invalid shape {key}={value}: {reason}")]
    InvalidShape {
        /// Shape key.
        key: String,
        /// Shape value.
        value: usize,
        /// Reason the shape is invalid.
        reason: String,
    },
    /// Invalid shape flag syntax.
    #[error("invalid shape flag {0:?}; expected K=V")]
    InvalidShapeSpec(String),
    /// Graph IR error.
    #[error(transparent)]
    Ir(#[from] IrError),
    /// Type inference error.
    #[error(transparent)]
    Type(#[from] TypeError),
    /// Graph surgery error.
    #[error(transparent)]
    Surgery(#[from] SurgeryError),
    /// egg search error.
    #[error(transparent)]
    Egg(#[from] EggError),
    /// Lowering error.
    #[error(transparent)]
    Lower(#[from] LowerError),
    /// Schedule legality error.
    #[error(transparent)]
    Legality(#[from] LegalityError),
    /// Schedule autotune error.
    #[error(transparent)]
    Tune(#[from] TuneError),
    /// C emission error.
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    /// C compile/run/parse error.
    #[error(transparent)]
    Bench(#[from] BenchError),
    /// Graph interpreter error.
    #[error(transparent)]
    Eval(#[from] EvalError),
    /// Loop interpreter error.
    #[error(transparent)]
    Loop(#[from] LoopError),
    /// Filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Self-check failed.
    #[error("self-check failed: {0}")]
    CheckFailed(String),
}

/// Parses a single `K=V` shape assignment.
///
/// # Errors
///
/// Returns [`PipelineError::InvalidShapeSpec`] if the string is malformed or
/// the value is not a `usize`.
pub fn parse_shape_spec(spec: &str) -> Result<(String, usize), PipelineError> {
    let Some((key, value)) = spec.split_once('=') else {
        return Err(PipelineError::InvalidShapeSpec(spec.to_owned()));
    };
    if key.is_empty() {
        return Err(PipelineError::InvalidShapeSpec(spec.to_owned()));
    }
    let value = value
        .parse::<usize>()
        .map_err(|_| PipelineError::InvalidShapeSpec(spec.to_owned()))?;
    Ok((key.to_owned(), value))
}

/// Runs the built-in example through rewrites, lowering, scheduling, codegen,
/// optional C execution, and optional self-checks.
///
/// # Errors
///
/// Returns [`PipelineError`] for any failed stage or self-check discrepancy.
#[allow(
    clippy::too_many_lines,
    reason = "the capstone pipeline is intentionally a linear stage-by-stage composition"
)]
pub fn run_pipeline(opts: &RunOpts) -> Result<RunReport, PipelineError> {
    let (graph, input_names) = build_example(&opts.example, &opts.shapes)?;
    infer_types(&graph)?;
    let rational_inputs = rational_inputs(&graph, opts.seed)?;
    let f64_inputs = f64_inputs(&graph, opts.seed)?;

    let mut notes = Vec::new();
    let mut verbose = Vec::new();
    if opts.verbose {
        verbose.push(format!("graph before optimization:\n{}", graph.to_text()));
    }

    let nodes_initial = graph.num_nodes();
    let rules = bit_exact_ruleset();
    let driver = run(
        &graph,
        &rules,
        &DriverConfig {
            mode: rewrite_mode(opts.mode),
            max_steps: 1_000,
        },
    )?;
    let rewrites = driver
        .applied
        .iter()
        .map(|applied| applied.rule.to_owned())
        .collect::<Vec<_>>();
    let nodes_after_driver = driver.graph.num_nodes();
    let mut optimized_graph = driver.graph;
    let driver_fixpoint = driver.reached_fixpoint;

    let mut egg_adopted = false;
    if opts.egg {
        let egg = vtc_rewrite::optimize_with_egg(
            &optimized_graph,
            &EggConfig {
                mode: rewrite_mode(opts.mode),
                node_limit: 10_000,
                iter_limit: 8,
                time_limit: Duration::from_millis(250),
                validation_trials: 32,
                seed: opts.seed,
            },
        )?;
        if egg.accepted && egg.result_nodes < optimized_graph.num_nodes() {
            optimized_graph = egg.graph;
            egg_adopted = true;
        } else {
            notes.push(format!(
                "egg returned accepted={} nodes {} -> {}; not adopted",
                egg.accepted, egg.original_nodes, egg.result_nodes
            ));
        }
    }
    let nodes_after_egg = optimized_graph.num_nodes();
    if opts.verbose {
        verbose.push(format!(
            "graph after optimization:\n{}",
            optimized_graph.to_text()
        ));
    }

    let lowered = lower(&optimized_graph)?;
    if opts.verbose {
        verbose.push(format!("kernel before scheduling:\n{}", lowered.to_text()));
    }

    let compiler_available = has_c_compiler();
    let used_cost = if opts.cost == CostKind::Measured && !compiler_available {
        notes.push("measured cost requested but no C compiler found; using static cost".to_owned());
        CostKind::Static
    } else {
        opts.cost
    };
    let tuned = tune_kernel(&lowered, opts, used_cost, &f64_inputs)?;
    if opts.verbose {
        verbose.push(format!(
            "kernel after scheduling:\n{}",
            tuned.kernel.to_text()
        ));
    }

    let emitted_c = emit_c(&tuned.kernel)?;
    if let Some(path) = &opts.emit {
        std::fs::write(path, &emitted_c)?;
    }

    let should_compile = !opts.no_compile && compiler_available;
    let compiled_outputs = if should_compile {
        Some(compile_and_run(&tuned.kernel, &f64_inputs)?)
    } else {
        if opts.no_compile {
            notes.push("C compilation skipped by --no-compile".to_owned());
        } else {
            notes.push("C compilation skipped because no C compiler was found".to_owned());
        }
        None
    };

    let check = if opts.check {
        check_pipeline(
            &graph,
            &tuned.kernel,
            opts.mode,
            &rational_inputs,
            &f64_inputs,
            compiled_outputs.as_ref(),
        )?
    } else {
        CheckStatus::NotRequested
    };

    let output_preview = compiled_outputs
        .as_deref()
        .map(preview_tensors)
        .unwrap_or_default();

    Ok(RunReport {
        example: opts.example.clone(),
        input_names,
        mode: opts.mode,
        egg_requested: opts.egg,
        requested_cost: opts.cost,
        used_cost,
        notes,
        nodes_initial,
        nodes_after_driver,
        nodes_after_egg,
        rewrites,
        driver_fixpoint,
        egg_adopted,
        schedule_moves: tuned.moves.iter().map(format_move).collect(),
        cost_before: tuned.cost_before,
        cost_after: tuned.cost_after,
        candidates_evaluated: tuned.candidates_evaluated,
        emitted_path: opts.emit.clone(),
        emitted_c,
        compiled: compiled_outputs.is_some(),
        output_preview,
        check,
        verbose,
    })
}

fn tune_kernel(
    kernel: &Kernel,
    opts: &RunOpts,
    used_cost: CostKind,
    f64_inputs: &HashMap<String, TensorF64>,
) -> Result<vtc_schedule::TuneResult, PipelineError> {
    let tune_cfg = TuneConfig {
        mode: opts.mode,
        tile_sizes: vec![2, 4],
        max_rounds: 4,
        validation_trials: 8,
        restarts: 0,
        seed: opts.seed,
    };
    let static_cost = StaticCost;
    let measured_cost = MeasuredCost {
        inputs: f64_inputs.clone(),
        reps: opts.reps.max(1),
    };
    let cost: &dyn CostModel = match used_cost {
        CostKind::Static => &static_cost,
        CostKind::Measured => &measured_cost,
    };
    autotune(kernel, cost, &tune_cfg).map_err(PipelineError::from)
}

fn check_pipeline(
    graph: &Graph,
    kernel: &Kernel,
    mode: Mode,
    rational_inputs: &HashMap<String, Tensor>,
    f64_inputs: &HashMap<String, TensorF64>,
    compiled_outputs: Option<&Vec<TensorF64>>,
) -> Result<CheckStatus, PipelineError> {
    let graph_exact = eval(graph, rational_inputs)?;
    let loop_exact = eval_loops(kernel, rational_inputs)?;
    if graph_exact != loop_exact {
        return Err(PipelineError::CheckFailed(
            "graph rational output differs from loop output".to_owned(),
        ));
    }

    let mut checks = vec!["rational graph == tuned loop".to_owned()];
    if mode == Mode::Strict {
        let graph_f64 = eval_f64(graph, f64_inputs)?;
        let loop_f64 = eval_loops_f64(kernel, f64_inputs)?;
        if !tensor_f64_slices_bit_eq(&graph_f64, &loop_f64) {
            return Err(PipelineError::CheckFailed(
                "graph f64 output differs from tuned loop output".to_owned(),
            ));
        }
        checks.push("strict f64 graph == tuned loop".to_owned());
        if let Some(compiled_outputs) = compiled_outputs {
            if !tensor_f64_slices_bit_eq(compiled_outputs, &loop_f64) {
                return Err(PipelineError::CheckFailed(
                    "compiled C output differs from tuned loop f64 output".to_owned(),
                ));
            }
            checks.push("compiled C == tuned loop f64".to_owned());
        }
    } else if let Some(compiled_outputs) = compiled_outputs {
        let loop_f64 = eval_loops_f64(kernel, f64_inputs)?;
        if !tensor_f64_slices_bit_eq(compiled_outputs, &loop_f64) {
            return Err(PipelineError::CheckFailed(
                "compiled C output differs from tuned loop f64 output".to_owned(),
            ));
        }
        checks.push("compiled C == tuned loop f64".to_owned());
    }
    Ok(CheckStatus::Passed(checks))
}

fn bit_exact_ruleset() -> RuleSet {
    let mut rules = RuleSet::new();
    rules.add(Box::new(NegNegElim));
    rules.add(Box::new(ReluIdempotentElim));
    rules.add(Box::new(ReshapeReshapeFuse));
    rules
}

fn rational_inputs(graph: &Graph, seed: u64) -> Result<HashMap<String, Tensor>, PipelineError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = HashMap::new();
    for (name, shape) in input_signature(graph)? {
        let numel = shape.numel().map_err(|_| PipelineError::InvalidShape {
            key: name.clone(),
            value: 0,
            reason: "shape element count overflow".to_owned(),
        })?;
        let values = (0..numel)
            .map(|_| rng.gen_range(-8_i64..=8_i64))
            .collect::<Vec<_>>();
        inputs.insert(name, Tensor::from_i64(shape, &values)?);
    }
    Ok(inputs)
}

fn f64_inputs(graph: &Graph, seed: u64) -> Result<HashMap<String, TensorF64>, PipelineError> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xf64f_64f6_4f64);
    let mut inputs = HashMap::new();
    for (name, shape) in input_signature(graph)? {
        let numel = shape.numel().map_err(|_| PipelineError::InvalidShape {
            key: name.clone(),
            value: 0,
            reason: "shape element count overflow".to_owned(),
        })?;
        let values = (0..numel)
            .map(|index| random_f64_value(index, &mut rng))
            .collect::<Vec<_>>();
        inputs.insert(name, TensorF64::from_f64(shape, &values)?);
    }
    Ok(inputs)
}

fn input_signature(graph: &Graph) -> Result<Vec<(String, vtc_ir::Shape)>, PipelineError> {
    graph
        .inputs()
        .iter()
        .map(|&input_id| {
            let node = graph.node(input_id)?;
            let Op::Input { name, ty } = node.op() else {
                return Err(PipelineError::CheckFailed(
                    "graph input list contains non-input node".to_owned(),
                ));
            };
            Ok((name.clone(), ty.shape().clone()))
        })
        .collect()
}

fn random_f64_value(index: usize, rng: &mut StdRng) -> f64 {
    if index.is_multiple_of(11) {
        return -0.0;
    }
    if index.is_multiple_of(17) {
        return 0.0;
    }
    let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
    let mantissa = f64::from(rng.gen_range(1_u16..=31));
    let exponent = rng.gen_range(-4..=4);
    sign * mantissa * 2.0_f64.powi(exponent)
}

fn tensor_f64_slices_bit_eq(left: &[TensorF64], right: &[TensorF64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.bit_eq(right))
}

fn preview_tensors(tensors: &[TensorF64]) -> Vec<String> {
    tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let values = tensor
                .data()
                .iter()
                .take(8)
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>();
            let suffix = if tensor.data().len() > 8 { ", ..." } else { "" };
            format!(
                "out{index} shape={} data=[{}{}]",
                tensor.shape(),
                values.join(", "),
                suffix
            )
        })
        .collect()
}

fn rewrite_mode(mode: Mode) -> RewriteMode {
    match mode {
        Mode::Strict => RewriteMode::Strict,
        Mode::FastMath => RewriteMode::FastMath,
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Strict => "strict",
        Mode::FastMath => "fastmath",
    }
}

fn format_move(desc: &MoveDesc) -> String {
    match desc {
        MoveDesc::Interchange { nest, level } => {
            format!("interchange(nest={nest},level={level})")
        }
        MoveDesc::Tile { nest, level, size } => {
            format!("tile(nest={nest},level={level},size={size})")
        }
        MoveDesc::Fuse { a, b } => format!("fuse({a},{b})"),
    }
}

fn list_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_owned()
    } else {
        items.join(", ")
    }
}
