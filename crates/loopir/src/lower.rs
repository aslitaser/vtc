//! Naive lowering from graph IR to affine loop IR.

use std::collections::HashSet;

use num_bigint::BigInt;
use num_rational::BigRational;
use thiserror::Error;
use vtc_ir::{
    DType, Graph, GraphTypes, IrError, NodeId, Op, Shape, TensorType, TypeError, infer_types,
};

use crate::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, LoopVar, ScalarExpr, Stmt,
};

/// Errors produced by graph-to-loop lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    /// Type inference rejected the graph.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// Lowering encountered an unsupported construct.
    #[error("unsupported lowering construct: {0}")]
    Unsupported(String),

    /// Internal consistency check failed.
    #[error("internal lowering error: {0}")]
    Internal(String),
}

impl From<IrError> for LowerError {
    fn from(error: IrError) -> Self {
        Self::Type(TypeError::from(error))
    }
}

/// Lowers a graph into a naive affine loop kernel.
///
/// Reshape is lowered as a free alias: the reshape node's result buffer is the
/// operand buffer, with no copy and no emitted statement.
///
/// # Errors
///
/// Returns [`LowerError`] if the graph is ill-typed or an internal lowering
/// consistency check fails.
pub fn lower(graph: &Graph) -> Result<Kernel, LowerError> {
    let types = infer_types(graph)?;
    let order = graph.topo_order().map_err(TypeError::from)?;
    let mut lowerer = Lowerer::default();
    let mut node_buffers = vec![None; graph.num_nodes()];

    for node_id in order {
        let node = graph.node(node_id)?;
        let ty = types
            .type_of(node_id)
            .ok_or_else(|| LowerError::Internal(format!("missing type for node {node_id}")))?;
        let buffer = lowerer.lower_node(node.op(), ty, &types, &node_buffers)?;
        set_node_buffer(&mut node_buffers, node_id, buffer)?;
    }

    let outputs = graph
        .outputs()
        .iter()
        .map(|&output| node_buffer(&node_buffers, output))
        .collect::<Result<Vec<_>, _>>()?;
    let output_shapes = graph
        .outputs()
        .iter()
        .map(|&output| type_shape(&types, output))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Kernel::new_with_output_shapes(
        lowerer.buffers,
        lowerer.body,
        lowerer.inputs,
        outputs,
        output_shapes,
    ))
}

#[derive(Debug, Default)]
struct Lowerer {
    buffers: Vec<Buffer>,
    body: Vec<Stmt>,
    inputs: Vec<BufferId>,
    next_loop_var: u32,
}

impl Lowerer {
    fn lower_node(
        &mut self,
        op: &Op,
        ty: &TensorType,
        types: &GraphTypes,
        node_buffers: &[Option<BufferId>],
    ) -> Result<BufferId, LowerError> {
        match op {
            Op::Input { name, .. } => {
                if ty.dtype() == DType::Bool {
                    return Err(LowerError::Unsupported("bool input".to_owned()));
                }
                let buffer = self.add_buffer(
                    name.clone(),
                    ty.shape().clone(),
                    BufferRole::Input(name.clone()),
                )?;
                self.inputs.push(buffer);
                Ok(buffer)
            }
            Op::Const { data, shape } => {
                if data.dtype() == DType::Bool {
                    return Err(LowerError::Unsupported("bool constant".to_owned()));
                }
                self.add_buffer(
                    "const".to_owned(),
                    shape.clone(),
                    BufferRole::Const(data.clone()),
                )
            }
            Op::Add(left, right) => self.lower_binary(
                ty.shape(),
                node_buffer(node_buffers, *left)?,
                node_buffer(node_buffers, *right)?,
                BinaryOp::Add,
            ),
            Op::Sub(left, right) => self.lower_binary(
                ty.shape(),
                node_buffer(node_buffers, *left)?,
                node_buffer(node_buffers, *right)?,
                BinaryOp::Sub,
            ),
            Op::Mul(left, right) => self.lower_binary(
                ty.shape(),
                node_buffer(node_buffers, *left)?,
                node_buffer(node_buffers, *right)?,
                BinaryOp::Mul,
            ),
            Op::Neg(input) => {
                self.lower_unary(ty.shape(), node_buffer(node_buffers, *input)?, UnaryOp::Neg)
            }
            Op::Relu(input) => self.lower_unary(
                ty.shape(),
                node_buffer(node_buffers, *input)?,
                UnaryOp::Relu,
            ),
            Op::Matmul(left, right) => {
                let left_shape = type_shape(types, *left)?;
                let right_shape = type_shape(types, *right)?;
                self.lower_matmul(
                    ty.shape(),
                    node_buffer(node_buffers, *left)?,
                    &left_shape,
                    node_buffer(node_buffers, *right)?,
                    &right_shape,
                )
            }
            Op::Sum {
                input,
                axes,
                keepdim,
            } => {
                let input_shape = type_shape(types, *input)?;
                self.lower_sum(
                    ty.shape(),
                    node_buffer(node_buffers, *input)?,
                    &input_shape,
                    axes,
                    *keepdim,
                )
            }
            Op::Reshape { input, .. } => node_buffer(node_buffers, *input),
        }
    }

    fn lower_binary(
        &mut self,
        shape: &Shape,
        left: BufferId,
        right: BufferId,
        op: BinaryOp,
    ) -> Result<BufferId, LowerError> {
        let output = self.add_temp(shape.clone())?;
        let vars = self.fresh_vars(shape.rank())?;
        let flat = row_major_affine(shape, &vars)?;
        let left = ScalarExpr::Load(BufferRef {
            buffer: left,
            index: flat.clone(),
        });
        let right = ScalarExpr::Load(BufferRef {
            buffer: right,
            index: flat.clone(),
        });
        let value = match op {
            BinaryOp::Add => ScalarExpr::Add(Box::new(left), Box::new(right)),
            BinaryOp::Sub => ScalarExpr::Sub(Box::new(left), Box::new(right)),
            BinaryOp::Mul => ScalarExpr::Mul(Box::new(left), Box::new(right)),
        };
        self.body.push(nest_for_shape(
            shape,
            &vars,
            Stmt::Assign {
                target: BufferRef {
                    buffer: output,
                    index: flat,
                },
                value,
            },
        )?);
        Ok(output)
    }

    fn lower_unary(
        &mut self,
        shape: &Shape,
        input: BufferId,
        op: UnaryOp,
    ) -> Result<BufferId, LowerError> {
        let output = self.add_temp(shape.clone())?;
        let vars = self.fresh_vars(shape.rank())?;
        let flat = row_major_affine(shape, &vars)?;
        let input = ScalarExpr::Load(BufferRef {
            buffer: input,
            index: flat.clone(),
        });
        let value = match op {
            UnaryOp::Neg => ScalarExpr::Neg(Box::new(input)),
            UnaryOp::Relu => ScalarExpr::Relu(Box::new(input)),
        };
        self.body.push(nest_for_shape(
            shape,
            &vars,
            Stmt::Assign {
                target: BufferRef {
                    buffer: output,
                    index: flat,
                },
                value,
            },
        )?);
        Ok(output)
    }

    fn lower_matmul(
        &mut self,
        output_shape: &Shape,
        left: BufferId,
        left_shape: &Shape,
        right: BufferId,
        right_shape: &Shape,
    ) -> Result<BufferId, LowerError> {
        let output = self.add_temp(output_shape.clone())?;
        let [m_dim, k_dim] = left_shape.dims() else {
            return Err(LowerError::Internal("matmul lhs is not rank 2".to_owned()));
        };
        let [_, n_dim] = right_shape.dims() else {
            return Err(LowerError::Internal("matmul rhs is not rank 2".to_owned()));
        };

        let m = LoopVar::new(self.take_loop_var()?);
        let n = LoopVar::new(self.take_loop_var()?);
        let k = LoopVar::new(self.take_loop_var()?);
        let out_index = AffineExpr::new(vec![(m, usize_to_i64(n_dim.get())?), (n, 1)], 0);
        let lhs_index = AffineExpr::new(vec![(m, usize_to_i64(k_dim.get())?), (k, 1)], 0);
        let rhs_index = AffineExpr::new(vec![(k, usize_to_i64(n_dim.get())?), (n, 1)], 0);

        let init = Stmt::Assign {
            target: BufferRef {
                buffer: output,
                index: out_index.clone(),
            },
            value: ScalarExpr::ConstScalar(BigRational::from_integer(BigInt::from(0))),
        };
        self.body
            .push(nest_loops(&[(m, m_dim.get()), (n, n_dim.get())], init)?);

        let acc = ScalarExpr::Add(
            Box::new(ScalarExpr::Load(BufferRef {
                buffer: output,
                index: out_index.clone(),
            })),
            Box::new(ScalarExpr::Mul(
                Box::new(ScalarExpr::Load(BufferRef {
                    buffer: left,
                    index: lhs_index,
                })),
                Box::new(ScalarExpr::Load(BufferRef {
                    buffer: right,
                    index: rhs_index,
                })),
            )),
        );
        self.body.push(nest_loops(
            &[(m, m_dim.get()), (n, n_dim.get()), (k, k_dim.get())],
            Stmt::Assign {
                target: BufferRef {
                    buffer: output,
                    index: out_index,
                },
                value: acc,
            },
        )?);
        Ok(output)
    }

    fn lower_sum(
        &mut self,
        output_shape: &Shape,
        input: BufferId,
        input_shape: &Shape,
        axes: &[usize],
        keepdim: bool,
    ) -> Result<BufferId, LowerError> {
        let output = self.add_temp(output_shape.clone())?;
        let out_vars = self.fresh_vars(output_shape.rank())?;
        let out_flat = row_major_affine(output_shape, &out_vars)?;
        self.body.push(nest_for_shape(
            output_shape,
            &out_vars,
            Stmt::Assign {
                target: BufferRef {
                    buffer: output,
                    index: out_flat,
                },
                value: ScalarExpr::ConstScalar(BigRational::from_integer(BigInt::from(0))),
            },
        )?);

        let in_vars = self.fresh_vars(input_shape.rank())?;
        let input_flat = row_major_affine(input_shape, &in_vars)?;
        let output_flat = sum_output_affine(input_shape, output_shape, axes, keepdim, &in_vars)?;
        let value = ScalarExpr::Add(
            Box::new(ScalarExpr::Load(BufferRef {
                buffer: output,
                index: output_flat.clone(),
            })),
            Box::new(ScalarExpr::Load(BufferRef {
                buffer: input,
                index: input_flat,
            })),
        );
        self.body.push(nest_for_shape(
            input_shape,
            &in_vars,
            Stmt::Assign {
                target: BufferRef {
                    buffer: output,
                    index: output_flat,
                },
                value,
            },
        )?);
        Ok(output)
    }

    fn add_temp(&mut self, shape: Shape) -> Result<BufferId, LowerError> {
        self.add_buffer("tmp".to_owned(), shape, BufferRole::Temp)
    }

    fn add_buffer(
        &mut self,
        name: String,
        shape: Shape,
        role: BufferRole,
    ) -> Result<BufferId, LowerError> {
        let id = BufferId::new(usize_to_u32(self.buffers.len())?);
        self.buffers.push(Buffer::new(id, name, shape, role));
        Ok(id)
    }

    fn fresh_vars(&mut self, rank: usize) -> Result<Vec<LoopVar>, LowerError> {
        (0..rank)
            .map(|_| self.take_loop_var().map(LoopVar::new))
            .collect()
    }

    fn take_loop_var(&mut self) -> Result<u32, LowerError> {
        let var = self.next_loop_var;
        self.next_loop_var = self
            .next_loop_var
            .checked_add(1)
            .ok_or_else(|| LowerError::Internal("loop variable id overflow".to_owned()))?;
        Ok(var)
    }
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Neg,
    Relu,
}

fn node_buffer(node_buffers: &[Option<BufferId>], node: NodeId) -> Result<BufferId, LowerError> {
    node_buffers
        .get(node.index())
        .and_then(|buffer| *buffer)
        .ok_or_else(|| LowerError::Internal(format!("missing lowered buffer for node {node}")))
}

fn set_node_buffer(
    node_buffers: &mut [Option<BufferId>],
    node: NodeId,
    buffer: BufferId,
) -> Result<(), LowerError> {
    let slot = node_buffers
        .get_mut(node.index())
        .ok_or_else(|| LowerError::Internal(format!("node {node} is out of range")))?;
    *slot = Some(buffer);
    Ok(())
}

fn type_shape(types: &GraphTypes, node: NodeId) -> Result<Shape, LowerError> {
    types
        .type_of(node)
        .map(|ty| ty.shape().clone())
        .ok_or_else(|| LowerError::Internal(format!("missing type for node {node}")))
}

fn nest_for_shape(shape: &Shape, vars: &[LoopVar], inner: Stmt) -> Result<Stmt, LowerError> {
    if shape.rank() != vars.len() {
        return Err(LowerError::Internal(
            "loop variable rank mismatch".to_owned(),
        ));
    }
    let loop_bounds = vars
        .iter()
        .copied()
        .zip(shape.dims().iter().map(|dim| dim.get()))
        .collect::<Vec<_>>();
    nest_loops(&loop_bounds, inner)
}

fn nest_loops(bounds: &[(LoopVar, usize)], inner: Stmt) -> Result<Stmt, LowerError> {
    let mut stmt = inner;
    for &(var, extent) in bounds.iter().rev() {
        stmt = Stmt::For {
            var,
            lo: AffineExpr::constant(0),
            hi: AffineExpr::constant(usize_to_i64(extent)?),
            body: vec![stmt],
        };
    }
    Ok(stmt)
}

fn row_major_affine(shape: &Shape, vars: &[LoopVar]) -> Result<AffineExpr, LowerError> {
    if shape.rank() != vars.len() {
        return Err(LowerError::Internal(
            "affine rank does not match variables".to_owned(),
        ));
    }

    let mut stride = 1usize;
    let mut terms = Vec::with_capacity(vars.len());
    for (&var, dim) in vars.iter().zip(shape.dims()).rev() {
        terms.push((var, usize_to_i64(stride)?));
        stride = stride
            .checked_mul(dim.get())
            .ok_or_else(|| LowerError::Internal("row-major stride overflow".to_owned()))?;
    }
    terms.reverse();
    Ok(AffineExpr::new(terms, 0))
}

fn sum_output_affine(
    input_shape: &Shape,
    output_shape: &Shape,
    axes: &[usize],
    keepdim: bool,
    input_vars: &[LoopVar],
) -> Result<AffineExpr, LowerError> {
    let reduced_axes: HashSet<_> = axes.iter().copied().collect();
    let output_vars = if keepdim {
        input_vars
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, var)| {
                if reduced_axes.contains(&axis) {
                    None
                } else {
                    Some(var)
                }
            })
            .collect::<Vec<_>>()
    } else {
        input_vars
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, var)| (!reduced_axes.contains(&axis)).then_some(Some(var)))
            .collect::<Vec<_>>()
    };

    if output_vars.len() != output_shape.rank() {
        return Err(LowerError::Internal(format!(
            "sum output rank mismatch: input {input_shape}, output {output_shape}"
        )));
    }

    let mut stride = 1usize;
    let mut terms = Vec::new();
    for (maybe_var, dim) in output_vars.iter().zip(output_shape.dims()).rev() {
        if let Some(var) = maybe_var {
            terms.push((*var, usize_to_i64(stride)?));
        }
        stride = stride
            .checked_mul(dim.get())
            .ok_or_else(|| LowerError::Internal("sum output stride overflow".to_owned()))?;
    }
    terms.reverse();
    Ok(AffineExpr::new(terms, 0))
}

fn usize_to_u32(value: usize) -> Result<u32, LowerError> {
    u32::try_from(value).map_err(|_| LowerError::Internal("id overflow".to_owned()))
}

fn usize_to_i64(value: usize) -> Result<i64, LowerError> {
    i64::try_from(value).map_err(|_| LowerError::Internal("extent overflow".to_owned()))
}
