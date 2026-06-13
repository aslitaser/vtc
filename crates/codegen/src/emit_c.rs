//! C source emission for affine loop kernels.

use std::fmt::Write as _;

use thiserror::Error;
use vtc_interp::TensorF64;
use vtc_loopir::{AffineExpr, BufferId, BufferRole, Kernel, LoopVar, ScalarExpr, Stmt};

/// Errors produced by C emission.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodegenError {
    /// The kernel contains a construct this backend cannot emit.
    #[error("unsupported codegen construct: {0}")]
    Unsupported(String),

    /// Internal consistency check failed.
    #[error("internal codegen error: {0}")]
    Internal(String),
}

/// Emits a self-contained C translation unit for `kernel`.
///
/// The generated program computes in `double`, emits `#pragma STDC FP_CONTRACT
/// OFF`, reads inputs as raw `uint64_t` bit patterns, and prints outputs as raw
/// `uint64_t` bit patterns. The kernel body follows the loop IR statement order
/// directly.
///
/// # Errors
///
/// Returns [`CodegenError`] if a buffer shape overflows, a constant cannot be
/// lifted into finite `double`, or a scalar literal is unsupported.
pub fn emit_c(kernel: &Kernel) -> Result<String, CodegenError> {
    let mut out = String::new();
    out.push_str("#pragma STDC FP_CONTRACT OFF\n");
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stdio.h>\n");
    out.push_str("#include <stdlib.h>\n");
    out.push_str("#include <string.h>\n\n");
    emit_kernel_function(kernel, &mut out)?;
    emit_main(kernel, &mut out)?;
    Ok(out)
}

fn emit_kernel_function(kernel: &Kernel, out: &mut String) -> Result<(), CodegenError> {
    out.push_str("static void vtc_kernel(");
    let mut params = Vec::new();
    for &input in kernel.inputs() {
        params.push(format!("const double *{}", buffer_name(input)));
    }
    for index in 0..kernel.outputs().len() {
        params.push(format!("double *out{index}"));
    }
    out.push_str(&params.join(", "));
    out.push_str(") {\n");

    for buffer in kernel.buffers() {
        match buffer.role() {
            BufferRole::Input(_) => {}
            BufferRole::Const(data) => {
                let tensor = TensorF64::from_data(data, buffer.shape()).map_err(|error| {
                    CodegenError::Unsupported(format!("constant cannot be lifted: {error}"))
                })?;
                emit_indent(out, 1);
                write!(
                    out,
                    "static const double {}[{}] = {{",
                    buffer_name(buffer.id()),
                    checked_numel(buffer.shape().numel())?
                )
                .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
                for (index, value) in tensor.data().iter().copied().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&double_literal(value)?);
                }
                out.push_str("};\n");
            }
            BufferRole::Temp | BufferRole::Output => {
                emit_indent(out, 1);
                writeln!(
                    out,
                    "double {}[{}];",
                    buffer_name(buffer.id()),
                    checked_numel(buffer.shape().numel())?
                )
                .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
            }
        }
    }

    for stmt in kernel.body() {
        emit_stmt(stmt, 1, out)?;
    }

    if kernel.outputs().len() != kernel.output_shapes().len() {
        return Err(CodegenError::Internal(
            "kernel output shape metadata length mismatch".to_owned(),
        ));
    }
    for (index, (&buffer, shape)) in kernel
        .outputs()
        .iter()
        .zip(kernel.output_shapes())
        .enumerate()
    {
        let numel = checked_numel(shape.numel())?;
        emit_indent(out, 1);
        writeln!(out, "for (long j = 0; j < {numel}; j++) {{")
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
        emit_indent(out, 2);
        writeln!(out, "out{index}[j] = {}[j];", buffer_name(buffer))
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
        emit_indent(out, 1);
        out.push_str("}\n");
    }

    out.push_str("}\n\n");
    Ok(())
}

fn emit_main(kernel: &Kernel, out: &mut String) -> Result<(), CodegenError> {
    out.push_str("int main(int argc, char **argv) {\n");
    emit_indent(out, 1);
    out.push_str("FILE *input = stdin;\n");
    emit_indent(out, 1);
    out.push_str("unsigned long reps = 1;\n");
    emit_indent(out, 1);
    out.push_str("if (argc > 1 && strcmp(argv[1], \"-\") != 0) {\n");
    emit_indent(out, 2);
    out.push_str("input = fopen(argv[1], \"r\");\n");
    emit_indent(out, 2);
    out.push_str("if (input == NULL) { return 2; }\n");
    emit_indent(out, 1);
    out.push_str("}\n");
    emit_indent(out, 1);
    out.push_str("if (argc > 2) { reps = strtoul(argv[2], NULL, 10); }\n");
    emit_indent(out, 1);
    out.push_str("if (reps == 0) { reps = 1; }\n");

    for &input in kernel.inputs() {
        let buffer = kernel
            .buffer(input)
            .ok_or_else(|| CodegenError::Internal(format!("missing input buffer {input}")))?;
        let numel = checked_numel(buffer.shape().numel())?;
        emit_indent(out, 1);
        writeln!(out, "double {}[{numel}];", buffer_name(input))
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
        emit_read_loop(input, numel, out)?;
    }

    for (index, shape) in kernel.output_shapes().iter().enumerate() {
        emit_indent(out, 1);
        writeln!(out, "double out{index}[{}];", checked_numel(shape.numel())?)
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
    }

    emit_indent(out, 1);
    out.push_str("for (unsigned long r = 0; r < reps; r++) {\n");
    emit_indent(out, 2);
    out.push_str("vtc_kernel(");
    let mut args = Vec::new();
    for &input in kernel.inputs() {
        args.push(buffer_name(input));
    }
    for index in 0..kernel.outputs().len() {
        args.push(format!("out{index}"));
    }
    out.push_str(&args.join(", "));
    out.push_str(");\n");
    emit_indent(out, 1);
    out.push_str("}\n");

    for (index, shape) in kernel.output_shapes().iter().enumerate() {
        emit_write_loop(index, checked_numel(shape.numel())?, out)?;
    }

    emit_indent(out, 1);
    out.push_str("if (input != stdin) { fclose(input); }\n");
    emit_indent(out, 1);
    out.push_str("return 0;\n");
    out.push_str("}\n");
    Ok(())
}

fn emit_read_loop(buffer: BufferId, numel: usize, out: &mut String) -> Result<(), CodegenError> {
    emit_indent(out, 1);
    writeln!(out, "for (long j = 0; j < {numel}; j++) {{")
        .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
    emit_indent(out, 2);
    out.push_str("unsigned long long bits = 0;\n");
    emit_indent(out, 2);
    out.push_str("if (fscanf(input, \"%llx\", &bits) != 1) { return 3; }\n");
    emit_indent(out, 2);
    writeln!(
        out,
        "memcpy(&{}[j], &bits, sizeof(double));",
        buffer_name(buffer)
    )
    .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
    emit_indent(out, 1);
    out.push_str("}\n");
    Ok(())
}

fn emit_write_loop(index: usize, numel: usize, out: &mut String) -> Result<(), CodegenError> {
    emit_indent(out, 1);
    writeln!(out, "for (long j = 0; j < {numel}; j++) {{")
        .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
    emit_indent(out, 2);
    out.push_str("unsigned long long bits = 0;\n");
    emit_indent(out, 2);
    writeln!(out, "memcpy(&bits, &out{index}[j], sizeof(double));")
        .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
    emit_indent(out, 2);
    out.push_str("printf(\"%016llx\\n\", bits);\n");
    emit_indent(out, 1);
    out.push_str("}\n");
    Ok(())
}

fn emit_stmt(stmt: &Stmt, indent: usize, out: &mut String) -> Result<(), CodegenError> {
    match stmt {
        Stmt::For { var, lo, hi, body } => {
            emit_indent(out, indent);
            writeln!(
                out,
                "for (long {} = {}; {} < {}; {}++) {{",
                var_name(*var),
                affine_text(lo),
                var_name(*var),
                affine_text(hi),
                var_name(*var)
            )
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
            for stmt in body {
                emit_stmt(stmt, indent.saturating_add(1), out)?;
            }
            emit_indent(out, indent);
            out.push_str("}\n");
        }
        Stmt::Assign { target, value } => {
            emit_indent(out, indent);
            writeln!(
                out,
                "{}[{}] = {};",
                buffer_name(target.buffer),
                affine_text(&target.index),
                scalar_text(value)?
            )
            .map_err(|_| CodegenError::Internal("format failed".to_owned()))?;
        }
    }
    Ok(())
}

fn scalar_text(value: &ScalarExpr) -> Result<String, CodegenError> {
    match value {
        ScalarExpr::Load(reference) => Ok(format!(
            "{}[{}]",
            buffer_name(reference.buffer),
            affine_text(&reference.index)
        )),
        ScalarExpr::ConstScalar(value) => {
            if value.denom().to_string() != "1" {
                return Err(CodegenError::Unsupported(
                    "non-integer scalar literal cannot be emitted exactly yet".to_owned(),
                ));
            }
            let integer = value.numer().to_string().parse::<i64>().map_err(|_| {
                CodegenError::Unsupported("scalar literal is outside supported range".to_owned())
            })?;
            #[allow(
                clippy::cast_precision_loss,
                reason = "codegen scalar literals are intentionally rounded to C double"
            )]
            let as_double = integer as f64;
            double_literal(as_double)
        }
        ScalarExpr::Add(left, right) => Ok(format!(
            "({} + {})",
            scalar_text(left)?,
            scalar_text(right)?
        )),
        ScalarExpr::Sub(left, right) => Ok(format!(
            "({} - {})",
            scalar_text(left)?,
            scalar_text(right)?
        )),
        ScalarExpr::Mul(left, right) => Ok(format!(
            "({} * {})",
            scalar_text(left)?,
            scalar_text(right)?
        )),
        ScalarExpr::Neg(input) => Ok(format!("(-{})", scalar_text(input)?)),
        ScalarExpr::Relu(input) => {
            let value = scalar_text(input)?;
            Ok(format!("(({value}) > 0.0 ? ({value}) : 0.0)"))
        }
    }
}

fn affine_text(expr: &AffineExpr) -> String {
    let mut parts = expr
        .terms()
        .iter()
        .filter(|(_, coeff)| *coeff != 0)
        .map(|(var, coeff)| match *coeff {
            1 => var_name(*var),
            -1 => format!("-{}", var_name(*var)),
            coeff => format!("{coeff}*{}", var_name(*var)),
        })
        .collect::<Vec<_>>();
    if expr.constant_term() != 0 || parts.is_empty() {
        parts.push(expr.constant_term().to_string());
    }
    parts.join(" + ")
}

fn checked_numel<E>(numel: Result<usize, E>) -> Result<usize, CodegenError> {
    let numel =
        numel.map_err(|_| CodegenError::Internal("shape element count overflow".to_owned()))?;
    if numel == 0 {
        Err(CodegenError::Unsupported(
            "zero-sized buffers are not supported by C backend".to_owned(),
        ))
    } else {
        Ok(numel)
    }
}

fn double_literal(value: f64) -> Result<String, CodegenError> {
    if !value.is_finite() {
        return Err(CodegenError::Unsupported(
            "non-finite constants are not supported".to_owned(),
        ));
    }
    let bits = value.to_bits();
    let sign = if (bits >> 63) == 1 { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0 && fraction == 0 {
        return Ok(format!("{sign}0x0p+0"));
    }
    if exponent == 0 {
        return Ok(format!("{sign}0x0.{fraction:013x}p-1022"));
    }
    Ok(format!("{sign}0x1.{fraction:013x}p{:+}", exponent - 1023))
}

fn buffer_name(id: BufferId) -> String {
    format!("b{}", id.id())
}

fn var_name(var: LoopVar) -> String {
    format!("i{}", var.id())
}

fn emit_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}
