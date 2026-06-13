//! Flat affine loop IR.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Write as _};

use num_rational::BigRational;
use vtc_ir::{Shape, TensorData};

/// A loop induction variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct LoopVar(u32);

impl LoopVar {
    /// Creates a loop variable from its stable numeric id.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the stable numeric id.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }
}

impl fmt::Display for LoopVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.0)
    }
}

/// An affine expression over loop variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineExpr {
    terms: Vec<(LoopVar, i64)>,
    constant: i64,
}

impl AffineExpr {
    /// Creates an affine expression from terms and a constant.
    #[must_use]
    pub fn new(terms: Vec<(LoopVar, i64)>, constant: i64) -> Self {
        Self { terms, constant }
    }

    /// Creates a constant affine expression.
    #[must_use]
    pub const fn constant(value: i64) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    /// Creates a single-variable affine expression.
    #[must_use]
    pub fn var(var: LoopVar) -> Self {
        Self {
            terms: vec![(var, 1)],
            constant: 0,
        }
    }

    /// Returns affine terms in source order.
    #[must_use]
    pub fn terms(&self) -> &[(LoopVar, i64)] {
        &self.terms
    }

    /// Returns the constant offset.
    #[must_use]
    pub const fn constant_term(&self) -> i64 {
        self.constant
    }

    /// Returns a new expression with `var` replaced by `replacement`.
    ///
    /// If `var` appears with coefficient `c`, the occurrence contributes
    /// `c * replacement` to the result. Like terms are combined and zero
    /// coefficients are removed, yielding a stable normalized term order.
    #[must_use]
    pub fn substitute(&self, var: LoopVar, replacement: &AffineExpr) -> Self {
        let mut terms = BTreeMap::new();
        let mut constant = self.constant;

        for (term_var, coeff) in &self.terms {
            if *term_var == var {
                constant = constant.saturating_add(coeff.saturating_mul(replacement.constant));
                for (replacement_var, replacement_coeff) in &replacement.terms {
                    add_term(
                        &mut terms,
                        *replacement_var,
                        coeff.saturating_mul(*replacement_coeff),
                    );
                }
            } else {
                add_term(&mut terms, *term_var, *coeff);
            }
        }

        Self::new(terms.into_iter().collect(), constant)
    }

    /// Evaluates this expression in a loop-variable environment.
    ///
    /// Missing variables are treated as zero so closed constant bounds and
    /// scalar rank-zero indices can share the same evaluator.
    #[must_use]
    pub fn eval(&self, env: &HashMap<LoopVar, i64>) -> i64 {
        self.terms.iter().fold(self.constant, |acc, (var, coeff)| {
            acc.saturating_add(coeff.saturating_mul(env.get(var).copied().unwrap_or(0)))
        })
    }
}

fn add_term(terms: &mut BTreeMap<LoopVar, i64>, var: LoopVar, coeff: i64) {
    if coeff == 0 {
        return;
    }
    let entry = terms.entry(var).or_insert(0);
    *entry = entry.saturating_add(coeff);
    if *entry == 0 {
        terms.remove(&var);
    }
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote = false;
        for (var, coeff) in &self.terms {
            if wrote {
                if *coeff >= 0 {
                    f.write_str(" + ")?;
                } else {
                    f.write_str(" - ")?;
                }
            } else if *coeff < 0 {
                f.write_str("-")?;
            }
            let abs = coeff.saturating_abs();
            if abs == 1 {
                write!(f, "{var}")?;
            } else {
                write!(f, "{abs}*{var}")?;
            }
            wrote = true;
        }
        if self.constant != 0 || !wrote {
            if wrote {
                if self.constant >= 0 {
                    write!(f, " + {}", self.constant)?;
                } else {
                    write!(f, " - {}", self.constant.saturating_abs())?;
                }
            } else {
                write!(f, "{}", self.constant)?;
            }
        }
        Ok(())
    }
}

/// A buffer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BufferId(u32);

impl BufferId {
    /// Creates a buffer id from its stable numeric id.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the stable numeric id.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }
}

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%b{}", self.0)
    }
}

/// Role of a kernel buffer.
#[derive(Debug, Clone)]
pub enum BufferRole {
    /// Named graph input.
    Input(String),
    /// Literal graph constant.
    Const(TensorData),
    /// Internal temporary.
    Temp,
    /// Graph output buffer.
    Output,
}

/// A flat row-major kernel buffer.
#[derive(Debug, Clone)]
pub struct Buffer {
    id: BufferId,
    name: String,
    shape: Shape,
    role: BufferRole,
}

impl Buffer {
    /// Creates a buffer.
    #[must_use]
    pub fn new(id: BufferId, name: String, shape: Shape, role: BufferRole) -> Self {
        Self {
            id,
            name,
            shape,
            role,
        }
    }

    /// Returns this buffer's id.
    #[must_use]
    pub const fn id(&self) -> BufferId {
        self.id
    }

    /// Returns this buffer's display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns this buffer's logical shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns this buffer's role.
    #[must_use]
    pub const fn role(&self) -> &BufferRole {
        &self.role
    }
}

/// A reference to a flat buffer element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferRef {
    /// Referenced buffer.
    pub buffer: BufferId,
    /// Flat affine row-major index.
    pub index: AffineExpr,
}

/// A scalar expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarExpr {
    /// Load one buffer element.
    Load(BufferRef),
    /// Exact rational scalar literal.
    ConstScalar(BigRational),
    /// Scalar addition.
    Add(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Scalar subtraction.
    Sub(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Scalar multiplication.
    Mul(Box<ScalarExpr>, Box<ScalarExpr>),
    /// Scalar negation.
    Neg(Box<ScalarExpr>),
    /// Scalar `ReLU`.
    Relu(Box<ScalarExpr>),
}

/// A loop IR statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// Counted loop with affine exclusive bounds.
    For {
        /// Induction variable.
        var: LoopVar,
        /// Inclusive lower bound.
        lo: AffineExpr,
        /// Exclusive upper bound.
        hi: AffineExpr,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// Scalar assignment.
    Assign {
        /// Target element.
        target: BufferRef,
        /// Value to store.
        value: ScalarExpr,
    },
}

/// A lowered affine loop kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    buffers: Vec<Buffer>,
    body: Vec<Stmt>,
    inputs: Vec<BufferId>,
    outputs: Vec<BufferId>,
    output_shapes: Vec<Shape>,
}

impl Kernel {
    /// Creates a kernel from buffers, statements, inputs, and outputs.
    #[must_use]
    pub fn new(
        buffers: Vec<Buffer>,
        body: Vec<Stmt>,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
    ) -> Self {
        let output_shapes = outputs
            .iter()
            .map(|id| {
                let Ok(index) = usize::try_from(id.id()) else {
                    return Shape::new(Vec::new());
                };
                buffers
                    .get(index)
                    .map_or_else(|| Shape::new(Vec::new()), |buffer| buffer.shape().clone())
            })
            .collect();
        Self::new_with_output_shapes(buffers, body, inputs, outputs, output_shapes)
    }

    /// Creates a kernel with explicit logical output shapes.
    ///
    /// This is used by lowering to represent reshape aliases: the output
    /// buffer id can be shared with its operand while the logical output shape
    /// follows the graph output type.
    #[must_use]
    pub fn new_with_output_shapes(
        buffers: Vec<Buffer>,
        body: Vec<Stmt>,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        output_shapes: Vec<Shape>,
    ) -> Self {
        Self {
            buffers,
            body,
            inputs,
            outputs,
            output_shapes,
        }
    }

    /// Returns all buffers.
    #[must_use]
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Returns kernel statements.
    #[must_use]
    pub fn body(&self) -> &[Stmt] {
        &self.body
    }

    /// Returns input buffer ids.
    #[must_use]
    pub fn inputs(&self) -> &[BufferId] {
        &self.inputs
    }

    /// Returns output buffer ids.
    #[must_use]
    pub fn outputs(&self) -> &[BufferId] {
        &self.outputs
    }

    /// Returns logical output shapes.
    #[must_use]
    pub fn output_shapes(&self) -> &[Shape] {
        &self.output_shapes
    }

    /// Returns a buffer by id.
    #[must_use]
    pub fn buffer(&self, id: BufferId) -> Option<&Buffer> {
        self.buffers.get(id.id() as usize)
    }

    /// Renders the kernel in a stable debug format.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("kernel {\n");
        out.push_str("  buffers:\n");
        for buffer in &self.buffers {
            out.push_str("    ");
            push_fmt(
                &mut out,
                format_args!(
                    "{} {}{} role={}\n",
                    buffer.id,
                    buffer.name,
                    buffer.shape,
                    role_text(buffer.role())
                ),
            );
        }
        out.push_str("  body:\n");
        for stmt in &self.body {
            format_stmt(stmt, 2, &mut out);
        }
        out.push_str("  outputs:");
        for output in &self.outputs {
            push_fmt(&mut out, format_args!(" {output}"));
        }
        out.push_str("\n}");
        out
    }
}

fn format_stmt(stmt: &Stmt, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    match stmt {
        Stmt::For { var, lo, hi, body } => {
            push_fmt(out, format_args!("{pad}for {var} in {lo}..{hi} {{\n"));
            for stmt in body {
                format_stmt(stmt, indent + 1, out);
            }
            push_fmt(out, format_args!("{pad}}}\n"));
        }
        Stmt::Assign { target, value } => {
            push_fmt(
                out,
                format_args!(
                    "{pad}{}[{}] = {}\n",
                    target.buffer,
                    target.index,
                    scalar_text(value)
                ),
            );
        }
    }
}

fn push_fmt(out: &mut String, args: fmt::Arguments<'_>) {
    if out.write_fmt(args).is_err() {
        out.push_str("<format-error>");
    }
}

fn scalar_text(value: &ScalarExpr) -> String {
    match value {
        ScalarExpr::Load(reference) => format!("{}[{}]", reference.buffer, reference.index),
        ScalarExpr::ConstScalar(value) => format!("{value}"),
        ScalarExpr::Add(left, right) => format!("({} + {})", scalar_text(left), scalar_text(right)),
        ScalarExpr::Sub(left, right) => format!("({} - {})", scalar_text(left), scalar_text(right)),
        ScalarExpr::Mul(left, right) => format!("({} * {})", scalar_text(left), scalar_text(right)),
        ScalarExpr::Neg(input) => format!("(-{})", scalar_text(input)),
        ScalarExpr::Relu(input) => format!("relu({})", scalar_text(input)),
    }
}

fn role_text(role: &BufferRole) -> String {
    match role {
        BufferRole::Input(name) => format!("input({name:?})"),
        BufferRole::Const(_) => "const".to_owned(),
        BufferRole::Temp => "temp".to_owned(),
        BufferRole::Output => "output".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{AffineExpr, LoopVar};

    #[test]
    fn substitute_replaces_scaled_variable() {
        let i = LoopVar::new(0);
        let io = LoopVar::new(1);
        let ii = LoopVar::new(2);
        let expr = AffineExpr::new(vec![(i, 3)], 5);
        let replacement = AffineExpr::new(vec![(io, 2), (ii, 1)], 0);

        assert_eq!(
            expr.substitute(i, &replacement),
            AffineExpr::new(vec![(io, 6), (ii, 3)], 5),
        );
    }

    #[test]
    fn substitute_absent_variable_is_identity() {
        let i = LoopVar::new(0);
        let j = LoopVar::new(1);
        let expr = AffineExpr::new(vec![(j, 7)], -3);
        let replacement = AffineExpr::new(vec![(LoopVar::new(2), 2)], 4);

        assert_eq!(expr.substitute(i, &replacement), expr);
    }

    #[test]
    fn substitute_combines_multi_term_expression() {
        let i = LoopVar::new(0);
        let j = LoopVar::new(1);
        let io = LoopVar::new(2);
        let ii = LoopVar::new(3);
        let expr = AffineExpr::new(vec![(i, 2), (j, 4), (io, 1)], 1);
        let replacement = AffineExpr::new(vec![(io, 3), (ii, -1)], 5);

        assert_eq!(
            expr.substitute(i, &replacement),
            AffineExpr::new(vec![(j, 4), (io, 7), (ii, -2)], 11),
        );
    }
}
