//! Compile, run, and measure emitted C kernels.

use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::hash::BuildHasher;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tempfile::TempDir;
use thiserror::Error;
use vtc_interp::TensorF64;
use vtc_loopir::{BufferRole, CostModel, Kernel};

use crate::emit_c::{CodegenError, emit_c};

/// Errors produced by C compilation, execution, or parsing.
#[derive(Debug, Error)]
pub enum BenchError {
    /// No C compiler was available.
    #[error("no C compiler found")]
    NoCompiler,

    /// C compilation failed.
    #[error("C compilation failed: {0}")]
    CompileFailed(String),

    /// The compiled program failed or produced invalid output.
    #[error("C run failed: {0}")]
    RunFailed(String),

    /// Filesystem or process I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Output parsing failed.
    #[error("parse error: {0}")]
    Parse(String),
}

impl From<CodegenError> for BenchError {
    fn from(error: CodegenError) -> Self {
        Self::CompileFailed(error.to_string())
    }
}

/// Returns true when a C compiler can be invoked.
#[must_use]
pub fn has_c_compiler() -> bool {
    detect_compiler().is_some()
}

/// Emits, compiles, and runs a kernel once.
///
/// Inputs and outputs are exchanged as raw `uint64_t` bit patterns so signed
/// zeros, NaNs, and infinities survive the Rust/C boundary exactly.
///
/// # Errors
///
/// Returns [`BenchError`] if no compiler is present, compilation fails, the
/// program exits unsuccessfully, output parsing fails, or input/output shapes
/// do not match the kernel.
pub fn compile_and_run<S: BuildHasher>(
    kernel: &Kernel,
    inputs: &HashMap<String, TensorF64, S>,
) -> Result<Vec<TensorF64>, BenchError> {
    let compiled = compile_kernel(kernel)?;
    let output = run_compiled(&compiled.binary, kernel, inputs, 1)?;
    parse_outputs(kernel, &output)
}

/// Measures the wall-clock duration of a compiled C kernel run.
///
/// The generated program executes the kernel `reps` times and prints the final
/// outputs. The measured duration includes process startup and output capture,
/// so it is a practical proxy rather than a pure kernel-cycle measurement.
///
/// # Errors
///
/// Returns [`BenchError`] on compilation, execution, or input validation
/// failure.
pub fn measure_runtime<S: BuildHasher>(
    kernel: &Kernel,
    inputs: &HashMap<String, TensorF64, S>,
    reps: usize,
) -> Result<Duration, BenchError> {
    let compiled = compile_kernel(kernel)?;
    let start = Instant::now();
    let _ = run_compiled(&compiled.binary, kernel, inputs, reps.max(1))?;
    Ok(start.elapsed())
}

/// Measured-runtime cost model for schedule autotuning.
///
/// Errors during compilation or execution return `u64::MAX`, making the
/// candidate unattractive without panicking from the infallible [`CostModel`]
/// API.
#[derive(Debug, Clone)]
pub struct MeasuredCost {
    /// Input assignment used for every measurement.
    pub inputs: HashMap<String, TensorF64>,
    /// Number of kernel repetitions per measurement.
    pub reps: usize,
}

impl CostModel for MeasuredCost {
    fn cost(&self, kernel: &Kernel) -> u64 {
        match measure_runtime(kernel, &self.inputs, self.reps) {
            Ok(duration) => u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            Err(_) => u64::MAX,
        }
    }
}

struct CompiledKernel {
    _dir: TempDir,
    binary: PathBuf,
}

fn compile_kernel(kernel: &Kernel) -> Result<CompiledKernel, BenchError> {
    let compiler = detect_compiler().ok_or(BenchError::NoCompiler)?;
    let source = emit_c(kernel)?;
    let dir = tempfile::tempdir()?;
    let source_path = dir.path().join("kernel.c");
    let binary_path = dir.path().join(binary_name());
    std::fs::write(&source_path, source)?;

    let output = Command::new(&compiler)
        .arg("-std=c11")
        .arg("-O2")
        .arg("-ffp-contract=off")
        .arg(&source_path)
        .arg("-o")
        .arg(&binary_path)
        .output()
        .map_err(|error| {
            if env::var_os("CC").is_some() {
                BenchError::CompileFailed(error.to_string())
            } else {
                BenchError::NoCompiler
            }
        })?;
    if !output.status.success() {
        return Err(BenchError::CompileFailed(stderr_text(&output.stderr)));
    }

    Ok(CompiledKernel {
        _dir: dir,
        binary: binary_path,
    })
}

fn run_compiled(
    binary: &Path,
    kernel: &Kernel,
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
    reps: usize,
) -> Result<String, BenchError> {
    let input_text = input_bits(kernel, inputs)?;
    let mut child = Command::new(binary)
        .arg("-")
        .arg(reps.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let Some(mut stdin) = child.stdin.take() else {
        return Err(BenchError::RunFailed(
            "failed to open child stdin".to_owned(),
        ));
    };
    stdin.write_all(input_text.as_bytes())?;
    drop(stdin);

    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(BenchError::RunFailed(stderr_text(&output.stderr)));
    }
    String::from_utf8(output.stdout).map_err(|error| BenchError::Parse(error.to_string()))
}

fn input_bits(
    kernel: &Kernel,
    inputs: &HashMap<String, TensorF64, impl BuildHasher>,
) -> Result<String, BenchError> {
    let mut out = String::new();
    for &input in kernel.inputs() {
        let buffer = kernel
            .buffer(input)
            .ok_or_else(|| BenchError::RunFailed(format!("missing input buffer {input}")))?;
        let BufferRole::Input(name) = buffer.role() else {
            return Err(BenchError::RunFailed(format!(
                "buffer {input} is not an input"
            )));
        };
        let tensor = inputs
            .get(name)
            .ok_or_else(|| BenchError::RunFailed(format!("missing input {name:?}")))?;
        if tensor.shape() != buffer.shape() {
            return Err(BenchError::RunFailed(format!(
                "input {name:?} shape mismatch: expected {}, got {}",
                buffer.shape(),
                tensor.shape()
            )));
        }
        for &value in tensor.data() {
            writeln!(&mut out, "{:016x}", value.to_bits())
                .map_err(|_| BenchError::RunFailed("failed to format input bits".to_owned()))?;
        }
    }
    Ok(out)
}

fn parse_outputs(kernel: &Kernel, stdout: &str) -> Result<Vec<TensorF64>, BenchError> {
    let expected = kernel
        .output_shapes()
        .iter()
        .map(|shape| checked_numel(shape.numel()))
        .collect::<Result<Vec<_>, _>>()?;
    let total = expected.iter().copied().sum::<usize>();
    let mut values = Vec::with_capacity(total);

    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let bits = u64::from_str_radix(line.trim(), 16)
            .map_err(|error| BenchError::Parse(error.to_string()))?;
        values.push(f64::from_bits(bits));
    }
    if values.len() != total {
        return Err(BenchError::Parse(format!(
            "expected {total} output values, got {}",
            values.len()
        )));
    }

    let mut cursor = 0usize;
    let mut outputs = Vec::with_capacity(expected.len());
    for (shape, &numel) in kernel.output_shapes().iter().zip(&expected) {
        let end = cursor.saturating_add(numel);
        let data = values
            .get(cursor..end)
            .ok_or_else(|| BenchError::Parse("output slice out of range".to_owned()))?
            .to_vec();
        outputs.push(
            TensorF64::from_f64(shape.clone(), &data)
                .map_err(|error| BenchError::Parse(error.to_string()))?,
        );
        cursor = end;
    }
    Ok(outputs)
}

fn detect_compiler() -> Option<String> {
    if let Some(compiler) = env::var_os("CC") {
        let compiler = compiler.to_string_lossy().into_owned();
        if compiler.is_empty() {
            return None;
        }
        return command_works(&compiler).then_some(compiler);
    }
    command_works("cc").then(|| "cc".to_owned())
}

fn command_works(compiler: &str) -> bool {
    Command::new(compiler)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn checked_numel<E>(numel: Result<usize, E>) -> Result<usize, BenchError> {
    let numel = numel.map_err(|_| BenchError::RunFailed("shape numel overflow".to_owned()))?;
    if numel == 0 {
        Err(BenchError::RunFailed(
            "zero-sized buffers are not supported".to_owned(),
        ))
    } else {
        Ok(numel)
    }
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "kernel.exe"
    } else {
        "kernel"
    }
}

fn stderr_text(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr).trim().to_owned()
}
