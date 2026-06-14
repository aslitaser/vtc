//! Command-line entry point for `vtc`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use vtc::{
    CostKind, PipelineError, RunOpts, build_example, list_examples, parse_shape_spec, run_pipeline,
};
use vtc_schedule::Mode;

#[derive(Debug, Parser)]
#[command(name = "vtc", version, about = "Verified tensor compiler placeholder")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List built-in example graphs.
    List,
    /// Run a built-in example through the full pipeline.
    Run(RunArgs),
}

#[derive(Debug, Parser)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "RunArgs mirrors explicit boolean clap flags"
)]
struct RunArgs {
    /// Built-in example name.
    #[arg(long, default_value = "matmul")]
    example: String,
    /// Shape parameter assignment. Accepts repeated flags or comma-separated K=V entries.
    #[arg(long = "shape", value_delimiter = ',')]
    shapes: Vec<String>,
    /// Numeric legality mode.
    #[arg(long, value_enum, default_value_t = CliMode::Strict)]
    mode: CliMode,
    /// Enable oracle-gated egg search.
    #[arg(long)]
    egg: bool,
    /// Schedule cost model.
    #[arg(long, value_enum, default_value_t = CliCost::Static)]
    cost: CliCost,
    /// Repetitions for measured C runtime.
    #[arg(long, default_value_t = 1)]
    reps: usize,
    /// Seed for deterministic generated inputs.
    #[arg(long, default_value_t = 0x18)]
    seed: u64,
    /// Run end-to-end self-checks.
    #[arg(long)]
    check: bool,
    /// Write emitted C to this path.
    #[arg(long)]
    emit: Option<PathBuf>,
    /// Skip C compilation and execution.
    #[arg(long)]
    no_compile: bool,
    /// Print graph and kernel snapshots.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    /// Strict IEEE-bit-preserving mode.
    Strict,
    /// Fast-math rational-only mode.
    Fastmath,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliCost {
    /// Static deterministic cost heuristic.
    Static,
    /// Measured C runtime cost when a compiler is available.
    Measured,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), PipelineError> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Run(RunArgs::default_run())) {
        Command::List => {
            for (name, shapes) in list_examples() {
                println!("{name}\t{shapes}");
            }
            Ok(())
        }
        Command::Run(args) => {
            let opts = args.to_opts()?;
            let report = run_pipeline(&opts)?;
            print!("{report}");
            Ok(())
        }
    }
}

impl RunArgs {
    fn default_run() -> Self {
        Self {
            example: "matmul".to_owned(),
            shapes: Vec::new(),
            mode: CliMode::Strict,
            egg: false,
            cost: CliCost::Static,
            reps: 1,
            seed: 0x18,
            check: false,
            emit: None,
            no_compile: false,
            verbose: false,
        }
    }

    fn to_opts(&self) -> Result<RunOpts, PipelineError> {
        let shapes = parse_shapes(&self.shapes)?;
        let opts = RunOpts {
            example: self.example.clone(),
            shapes,
            mode: self.mode.into(),
            egg: self.egg,
            cost: self.cost.into(),
            reps: self.reps,
            seed: self.seed,
            check: self.check,
            emit: self.emit.clone(),
            no_compile: self.no_compile,
            verbose: self.verbose,
        };
        let _ = build_example(&opts.example, &opts.shapes)?;
        Ok(opts)
    }
}

impl From<CliMode> for Mode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Strict => Self::Strict,
            CliMode::Fastmath => Self::FastMath,
        }
    }
}

impl From<CliCost> for CostKind {
    fn from(value: CliCost) -> Self {
        match value {
            CliCost::Static => Self::Static,
            CliCost::Measured => Self::Measured,
        }
    }
}

fn parse_shapes(values: &[String]) -> Result<HashMap<String, usize>, PipelineError> {
    let mut shapes = HashMap::new();
    for value in values {
        let (key, parsed) = parse_shape_spec(value)?;
        shapes.insert(key, parsed);
    }
    Ok(shapes)
}
