//! egg language and graph conversion utilities.

#![allow(
    clippy::module_name_repetitions,
    reason = "step-10 public API names intentionally include Egg and AtomTable"
)]

use std::collections::HashMap;

use egg::{Id, RecExpr, define_language};
use thiserror::Error;
use vtc_ir::{Graph, GraphBuilder, IrError, NodeId, Op, TypeError, infer_types};

define_language! {
    /// Algebraic egg language used for untrusted equality-saturation search.
    #[allow(
        missing_docs,
        reason = "egg's define_language macro does not accept per-variant doc comments"
    )]
    pub enum EggLang {
        Atom(u32),
        "neg" = Neg([Id; 1]),
        "relu" = Relu([Id; 1]),
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "matmul" = Matmul([Id; 2]),
    }
}

/// Errors produced by egg conversion, search, and validation.
#[derive(Debug, Error)]
pub enum EggError {
    /// Conversion between graph IR and egg expressions failed.
    #[error("egg conversion failed: {0}")]
    ConversionFailed(String),

    /// Back-conversion produced an invalid graph.
    #[error("egg back-conversion produced invalid graph: {0}")]
    BackConversionInvalid(String),

    /// Internal consistency check failed.
    #[error("internal egg error: {0}")]
    Internal(String),
}

impl From<IrError> for EggError {
    fn from(error: IrError) -> Self {
        Self::BackConversionInvalid(error.to_string())
    }
}

impl From<TypeError> for EggError {
    fn from(error: TypeError) -> Self {
        Self::BackConversionInvalid(error.to_string())
    }
}

/// Side table mapping egg atoms back to original graph nodes.
#[derive(Debug, Clone, Default)]
pub struct AtomTable {
    nodes: Vec<NodeId>,
    indices: HashMap<NodeId, u32>,
}

impl AtomTable {
    /// Creates an empty atom table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the original node id for an atom index.
    ///
    /// # Errors
    ///
    /// Returns [`EggError::ConversionFailed`] if the atom index is unknown.
    pub fn node(&self, atom: u32) -> Result<NodeId, EggError> {
        self.nodes
            .get(atom as usize)
            .copied()
            .ok_or_else(|| EggError::ConversionFailed(format!("unknown atom index {atom}")))
    }

    fn atom_for(&mut self, node: NodeId) -> Result<u32, EggError> {
        if let Some(&index) = self.indices.get(&node) {
            return Ok(index);
        }
        let index = u32::try_from(self.nodes.len())
            .map_err(|_| EggError::ConversionFailed("too many atom nodes".to_owned()))?;
        self.nodes.push(node);
        self.indices.insert(node, index);
        Ok(index)
    }
}

/// Converts a graph output into an egg expression plus atom side table.
///
/// Input, const, sum, and reshape roots become opaque atoms. Algebraic core ops
/// are represented directly. Shared IR nodes are memoized into one egg id.
///
/// # Errors
///
/// Returns [`EggError`] if graph traversal fails.
pub fn graph_to_recexpr(
    graph: &Graph,
    output: NodeId,
) -> Result<(RecExpr<EggLang>, AtomTable), EggError> {
    let mut expr = RecExpr::default();
    let mut atoms = AtomTable::new();
    let mut memo = HashMap::new();
    emit_recexpr_node(graph, output, &mut expr, &mut atoms, &mut memo)?;
    Ok((expr, atoms))
}

/// Converts one extracted egg expression back to a validated graph.
///
/// Atoms splice the original IR sub-DAG rooted at their recorded node. The
/// rebuilt graph is structurally validated and type-checked before returning.
///
/// # Errors
///
/// Returns [`EggError`] if the expression references unknown atoms, cannot be
/// represented as graph IR, or rebuild validation fails.
pub fn recexpr_to_graph(
    expr: &RecExpr<EggLang>,
    atoms: &AtomTable,
    original: &Graph,
) -> Result<Graph, EggError> {
    recexprs_to_graph(&[(expr, atoms)], original)
}

pub(crate) fn recexprs_to_graph(
    outputs: &[(&RecExpr<EggLang>, &AtomTable)],
    original: &Graph,
) -> Result<Graph, EggError> {
    let mut builder = GraphBuilder::new();
    let mut atom_memo = HashMap::new();
    let mut new_outputs = Vec::with_capacity(outputs.len());

    for &(expr, atoms) in outputs {
        let root = root_id(expr)?;
        let mut expr_memo = HashMap::new();
        let output = emit_graph_expr(
            expr,
            atoms,
            root,
            original,
            &mut builder,
            &mut expr_memo,
            &mut atom_memo,
        )?;
        new_outputs.push(output);
    }

    for output in new_outputs {
        builder.mark_output(output)?;
    }
    let graph = builder.build()?;
    infer_types(&graph)?;
    Ok(graph)
}

fn emit_recexpr_node(
    graph: &Graph,
    node: NodeId,
    expr: &mut RecExpr<EggLang>,
    atoms: &mut AtomTable,
    memo: &mut HashMap<NodeId, Id>,
) -> Result<Id, EggError> {
    if let Some(&id) = memo.get(&node) {
        return Ok(id);
    }

    let op = graph.node(node)?.op();
    let id = match op {
        Op::Neg(input) => {
            let input = emit_recexpr_node(graph, *input, expr, atoms, memo)?;
            expr.add(EggLang::Neg([input]))
        }
        Op::Relu(input) => {
            let input = emit_recexpr_node(graph, *input, expr, atoms, memo)?;
            expr.add(EggLang::Relu([input]))
        }
        Op::Add(left, right) => {
            let left = emit_recexpr_node(graph, *left, expr, atoms, memo)?;
            let right = emit_recexpr_node(graph, *right, expr, atoms, memo)?;
            expr.add(EggLang::Add([left, right]))
        }
        Op::Sub(left, right) => {
            let left = emit_recexpr_node(graph, *left, expr, atoms, memo)?;
            let right = emit_recexpr_node(graph, *right, expr, atoms, memo)?;
            expr.add(EggLang::Sub([left, right]))
        }
        Op::Mul(left, right) => {
            let left = emit_recexpr_node(graph, *left, expr, atoms, memo)?;
            let right = emit_recexpr_node(graph, *right, expr, atoms, memo)?;
            expr.add(EggLang::Mul([left, right]))
        }
        Op::Matmul(left, right) => {
            let left = emit_recexpr_node(graph, *left, expr, atoms, memo)?;
            let right = emit_recexpr_node(graph, *right, expr, atoms, memo)?;
            expr.add(EggLang::Matmul([left, right]))
        }
        Op::Input { .. } | Op::Const { .. } | Op::Sum { .. } | Op::Reshape { .. } => {
            let atom = atoms.atom_for(node)?;
            expr.add(EggLang::Atom(atom))
        }
    };
    memo.insert(node, id);
    Ok(id)
}

fn emit_graph_expr(
    expr: &RecExpr<EggLang>,
    atoms: &AtomTable,
    id: Id,
    original: &Graph,
    builder: &mut GraphBuilder,
    expr_memo: &mut HashMap<Id, NodeId>,
    atom_memo: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, EggError> {
    if let Some(&node) = expr_memo.get(&id) {
        return Ok(node);
    }

    let enode = expr
        .as_ref()
        .get(usize::from(id))
        .ok_or_else(|| EggError::ConversionFailed(format!("missing expr id {id:?}")))?;
    let node = match enode {
        EggLang::Atom(atom) => {
            let original_node = atoms.node(*atom)?;
            copy_original_subdag(original, original_node, builder, atom_memo)?
        }
        EggLang::Neg([input]) => {
            let input =
                emit_graph_expr(expr, atoms, *input, original, builder, expr_memo, atom_memo)?;
            builder.neg(input)?
        }
        EggLang::Relu([input]) => {
            let input =
                emit_graph_expr(expr, atoms, *input, original, builder, expr_memo, atom_memo)?;
            builder.relu(input)?
        }
        EggLang::Add([left, right]) => {
            let left =
                emit_graph_expr(expr, atoms, *left, original, builder, expr_memo, atom_memo)?;
            let right =
                emit_graph_expr(expr, atoms, *right, original, builder, expr_memo, atom_memo)?;
            builder.add(left, right)?
        }
        EggLang::Sub([left, right]) => {
            let left =
                emit_graph_expr(expr, atoms, *left, original, builder, expr_memo, atom_memo)?;
            let right =
                emit_graph_expr(expr, atoms, *right, original, builder, expr_memo, atom_memo)?;
            builder.sub(left, right)?
        }
        EggLang::Mul([left, right]) => {
            let left =
                emit_graph_expr(expr, atoms, *left, original, builder, expr_memo, atom_memo)?;
            let right =
                emit_graph_expr(expr, atoms, *right, original, builder, expr_memo, atom_memo)?;
            builder.mul(left, right)?
        }
        EggLang::Matmul([left, right]) => {
            let left =
                emit_graph_expr(expr, atoms, *left, original, builder, expr_memo, atom_memo)?;
            let right =
                emit_graph_expr(expr, atoms, *right, original, builder, expr_memo, atom_memo)?;
            builder.matmul(left, right)?
        }
    };
    expr_memo.insert(id, node);
    Ok(node)
}

fn copy_original_subdag(
    original: &Graph,
    node: NodeId,
    builder: &mut GraphBuilder,
    memo: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, EggError> {
    if let Some(&new_node) = memo.get(&node) {
        return Ok(new_node);
    }

    let op = original.node(node)?.op();
    let new_node = match op {
        Op::Input { name, ty } => builder.input(name.clone(), ty.dtype(), ty.shape().clone())?,
        Op::Const { data, shape } => builder.constant(data.clone(), shape.clone())?,
        Op::Add(left, right) => {
            let left = copy_original_subdag(original, *left, builder, memo)?;
            let right = copy_original_subdag(original, *right, builder, memo)?;
            builder.add(left, right)?
        }
        Op::Sub(left, right) => {
            let left = copy_original_subdag(original, *left, builder, memo)?;
            let right = copy_original_subdag(original, *right, builder, memo)?;
            builder.sub(left, right)?
        }
        Op::Mul(left, right) => {
            let left = copy_original_subdag(original, *left, builder, memo)?;
            let right = copy_original_subdag(original, *right, builder, memo)?;
            builder.mul(left, right)?
        }
        Op::Neg(input) => {
            let input = copy_original_subdag(original, *input, builder, memo)?;
            builder.neg(input)?
        }
        Op::Relu(input) => {
            let input = copy_original_subdag(original, *input, builder, memo)?;
            builder.relu(input)?
        }
        Op::Matmul(left, right) => {
            let left = copy_original_subdag(original, *left, builder, memo)?;
            let right = copy_original_subdag(original, *right, builder, memo)?;
            builder.matmul(left, right)?
        }
        Op::Sum {
            input,
            axes,
            keepdim,
        } => {
            let input = copy_original_subdag(original, *input, builder, memo)?;
            builder.sum(input, axes.clone(), *keepdim)?
        }
        Op::Reshape { input, new_shape } => {
            let input = copy_original_subdag(original, *input, builder, memo)?;
            builder.reshape(input, new_shape.clone())?
        }
    };
    memo.insert(node, new_node);
    Ok(new_node)
}

fn root_id(expr: &RecExpr<EggLang>) -> Result<Id, EggError> {
    expr.as_ref()
        .len()
        .checked_sub(1)
        .map(Id::from)
        .ok_or_else(|| EggError::ConversionFailed("empty RecExpr".to_owned()))
}
