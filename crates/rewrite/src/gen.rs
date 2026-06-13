//! Well-typed random graph generation for rewrite differential tests.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vtc_interp::Tensor;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorData, TensorType};

const MAX_RANK: usize = 3;
const INITIAL_LEAVES: usize = 2;

/// Configuration for random graph and input generation.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Numeric dtype used for every generated graph value.
    pub dtype: DType,
    /// Approximate number of operation nodes to add after initial leaves.
    pub size: usize,
    /// Maximum generated dimension extent.
    pub max_dim: usize,
    /// Inclusive absolute bound for generated input and constant values.
    pub value_range: i64,
    /// Base seed used by deterministic test harnesses.
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            dtype: DType::I64,
            size: 8,
            max_dim: 4,
            value_range: 8,
            seed: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

/// Pattern guaranteed to occur in a generated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// Nested negation, `neg(neg(x))`.
    NegNeg,
    /// Nested `ReLU`, `relu(relu(x))`.
    ReluRelu,
    /// Nested reshape, `reshape(reshape(x))`.
    ReshapeReshape,
}

#[derive(Debug, Clone)]
struct Value {
    id: NodeId,
    ty: TensorType,
}

/// Generates a random well-typed graph and its input names.
#[must_use]
pub fn random_graph(cfg: &GenConfig, rng: &mut StdRng) -> (Graph, Vec<String>) {
    generate_until_ok(cfg, rng, None)
}

/// Generates a random well-typed graph containing `pattern` and its input names.
#[must_use]
pub fn graph_with_pattern(
    cfg: &GenConfig,
    rng: &mut StdRng,
    pattern: Pattern,
) -> (Graph, Vec<String>) {
    generate_until_ok(cfg, rng, Some(pattern))
}

/// Generates random interpreter inputs for a graph's declared inputs.
///
/// Generated values are small and bounded. Integer dtypes use exact integer
/// tensors; floating dtypes use finite dyadic `f64` values.
#[must_use]
pub fn random_inputs(graph: &Graph, value_range: i64, rng: &mut StdRng) -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let Ok(node) = graph.node(input_id) else {
            continue;
        };
        let Op::Input { name, ty } = node.op() else {
            continue;
        };
        let Some(numel) = ty.shape().numel().ok() else {
            continue;
        };
        let tensor = match ty.dtype() {
            DType::F32 | DType::F64 => {
                let values = (0..numel)
                    .map(|_| bounded_dyadic(value_range, rng))
                    .collect::<Vec<_>>();
                Tensor::from_f64(ty.shape().clone(), &values)
            }
            DType::I32 | DType::I64 => {
                let values = (0..numel)
                    .map(|_| bounded_i64(value_range, rng))
                    .collect::<Vec<_>>();
                Tensor::from_i64(ty.shape().clone(), &values)
            }
            DType::Bool => continue,
        };
        if let Ok(tensor) = tensor {
            inputs.insert(name.clone(), tensor);
        }
    }
    inputs
}

fn generate_until_ok(
    cfg: &GenConfig,
    rng: &mut StdRng,
    pattern: Option<Pattern>,
) -> (Graph, Vec<String>) {
    loop {
        if let Ok(result) = try_generated_graph(cfg, rng, pattern) {
            return result;
        }
        let retry_seed = rng.r#gen::<u64>();
        *rng = StdRng::seed_from_u64(retry_seed);
    }
}

fn try_generated_graph(
    cfg: &GenConfig,
    rng: &mut StdRng,
    pattern: Option<Pattern>,
) -> Result<(Graph, Vec<String>), vtc_ir::IrError> {
    let mut builder = GraphBuilder::new();
    let mut input_names = Vec::new();
    let mut pool = Vec::new();

    for _ in 0..INITIAL_LEAVES {
        let shape = random_shape(cfg, rng);
        let value = append_leaf(&mut builder, &mut input_names, cfg, rng, shape)?;
        pool.push(value);
    }

    for _ in 0..cfg.size {
        let value = append_random_value(&mut builder, &mut input_names, &pool, cfg, rng)?;
        pool.push(value);
    }

    let output = match pattern {
        Some(pattern) => {
            let value = append_pattern(&mut builder, &pool, rng, pattern)?;
            pool.push(value.clone());
            value.id
        }
        None => pool[random_index(pool.len(), rng)].id,
    };

    builder.mark_output(output)?;
    Ok((builder.build()?, input_names))
}

fn append_random_value(
    builder: &mut GraphBuilder,
    input_names: &mut Vec<String>,
    pool: &[Value],
    cfg: &GenConfig,
    rng: &mut StdRng,
) -> Result<Value, vtc_ir::IrError> {
    match rng.gen_range(0..9) {
        0 => {
            let shape = random_shape(cfg, rng);
            append_leaf(builder, input_names, cfg, rng, shape)
        }
        1 => append_binary(builder, input_names, pool, cfg, rng, BinaryOp::Add),
        2 => append_binary(builder, input_names, pool, cfg, rng, BinaryOp::Sub),
        3 => append_binary(builder, input_names, pool, cfg, rng, BinaryOp::Mul),
        4 => append_unary(builder, pool, rng, UnaryOp::Neg),
        5 => append_unary(builder, pool, rng, UnaryOp::Relu),
        6 => append_matmul(builder, input_names, pool, cfg, rng),
        7 => append_sum(builder, pool, rng),
        _ => append_reshape(builder, pool, rng),
    }
}

fn append_pattern(
    builder: &mut GraphBuilder,
    pool: &[Value],
    rng: &mut StdRng,
    pattern: Pattern,
) -> Result<Value, vtc_ir::IrError> {
    match pattern {
        Pattern::NegNeg => {
            let value = pool[random_index(pool.len(), rng)].clone();
            let first = builder.neg(value.id)?;
            let second = builder.neg(first)?;
            Ok(Value {
                id: second,
                ty: value.ty,
            })
        }
        Pattern::ReluRelu => {
            let value = pool[random_index(pool.len(), rng)].clone();
            let first = builder.relu(value.id)?;
            let second = builder.relu(first)?;
            Ok(Value {
                id: second,
                ty: value.ty,
            })
        }
        Pattern::ReshapeReshape => {
            let value = pool[random_index(pool.len(), rng)].clone();
            let first_shape = random_reshape_shape(value.ty.shape(), rng);
            let second_shape = random_reshape_shape(value.ty.shape(), rng);
            let first = builder.reshape(value.id, first_shape)?;
            let second = builder.reshape(first, second_shape.clone())?;
            Ok(Value {
                id: second,
                ty: TensorType::new(value.ty.dtype(), second_shape),
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

fn append_binary(
    builder: &mut GraphBuilder,
    input_names: &mut Vec<String>,
    pool: &[Value],
    cfg: &GenConfig,
    rng: &mut StdRng,
    op: BinaryOp,
) -> Result<Value, vtc_ir::IrError> {
    let left = pool[random_index(pool.len(), rng)].clone();
    let candidates = pool
        .iter()
        .filter(|value| value.ty.shape() == left.ty.shape())
        .cloned()
        .collect::<Vec<_>>();
    let right = if candidates.is_empty() {
        append_leaf_of_type(builder, input_names, cfg, rng, &left.ty)?
    } else {
        candidates[random_index(candidates.len(), rng)].clone()
    };

    let id = match op {
        BinaryOp::Add => builder.add(left.id, right.id)?,
        BinaryOp::Sub => builder.sub(left.id, right.id)?,
        BinaryOp::Mul => builder.mul(left.id, right.id)?,
    };
    Ok(Value { id, ty: left.ty })
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Neg,
    Relu,
}

fn append_unary(
    builder: &mut GraphBuilder,
    pool: &[Value],
    rng: &mut StdRng,
    op: UnaryOp,
) -> Result<Value, vtc_ir::IrError> {
    let input = pool[random_index(pool.len(), rng)].clone();
    let id = match op {
        UnaryOp::Neg => builder.neg(input.id)?,
        UnaryOp::Relu => builder.relu(input.id)?,
    };
    Ok(Value { id, ty: input.ty })
}

fn append_matmul(
    builder: &mut GraphBuilder,
    input_names: &mut Vec<String>,
    pool: &[Value],
    cfg: &GenConfig,
    rng: &mut StdRng,
) -> Result<Value, vtc_ir::IrError> {
    let rank_two = pool
        .iter()
        .filter(|value| value.ty.shape().rank() == 2)
        .cloned()
        .collect::<Vec<_>>();
    let left = if rank_two.is_empty() {
        let shape = random_matrix_shape(cfg, rng);
        append_leaf(builder, input_names, cfg, rng, shape)?
    } else {
        rank_two[random_index(rank_two.len(), rng)].clone()
    };

    let k = left.ty.shape().dims()[1].get();
    let n = random_dim(cfg, rng);
    let right_shape = Shape::new(vec![Dim::new(k), Dim::new(n)]);
    let right = append_leaf(builder, input_names, cfg, rng, right_shape)?;
    let id = builder.matmul(left.id, right.id)?;
    let output_shape = Shape::new(vec![left.ty.shape().dims()[0], Dim::new(n)]);
    Ok(Value {
        id,
        ty: TensorType::new(numeric_dtype(cfg.dtype), output_shape),
    })
}

fn append_sum(
    builder: &mut GraphBuilder,
    pool: &[Value],
    rng: &mut StdRng,
) -> Result<Value, vtc_ir::IrError> {
    let input = pool[random_index(pool.len(), rng)].clone();
    let axes = random_axes(input.ty.shape().rank(), rng);
    let keepdim = rng.gen_bool(0.5);
    let output_shape = sum_output_shape(input.ty.shape(), &axes, keepdim);
    let id = builder.sum(input.id, axes, keepdim)?;
    Ok(Value {
        id,
        ty: TensorType::new(input.ty.dtype(), output_shape),
    })
}

fn append_reshape(
    builder: &mut GraphBuilder,
    pool: &[Value],
    rng: &mut StdRng,
) -> Result<Value, vtc_ir::IrError> {
    let input = pool[random_index(pool.len(), rng)].clone();
    let new_shape = random_reshape_shape(input.ty.shape(), rng);
    let id = builder.reshape(input.id, new_shape.clone())?;
    Ok(Value {
        id,
        ty: TensorType::new(input.ty.dtype(), new_shape),
    })
}

fn append_leaf(
    builder: &mut GraphBuilder,
    input_names: &mut Vec<String>,
    cfg: &GenConfig,
    rng: &mut StdRng,
    shape: Shape,
) -> Result<Value, vtc_ir::IrError> {
    let ty = TensorType::new(numeric_dtype(cfg.dtype), shape);
    append_leaf_of_type(builder, input_names, cfg, rng, &ty)
}

fn append_leaf_of_type(
    builder: &mut GraphBuilder,
    input_names: &mut Vec<String>,
    cfg: &GenConfig,
    rng: &mut StdRng,
    ty: &TensorType,
) -> Result<Value, vtc_ir::IrError> {
    if rng.gen_bool(0.5) {
        let name = format!("x{}", input_names.len());
        let id = builder.input(name.clone(), ty.dtype(), ty.shape().clone())?;
        input_names.push(name);
        Ok(Value { id, ty: ty.clone() })
    } else {
        append_random_const(builder, cfg, rng, ty)
    }
}

fn append_random_const(
    builder: &mut GraphBuilder,
    cfg: &GenConfig,
    rng: &mut StdRng,
    ty: &TensorType,
) -> Result<Value, vtc_ir::IrError> {
    let data = tensor_data_for_shape(ty.dtype(), ty.shape(), cfg.value_range, rng)?;
    let id = builder.constant(data, ty.shape().clone())?;
    Ok(Value { id, ty: ty.clone() })
}

fn tensor_data_for_shape(
    dtype: DType,
    shape: &Shape,
    value_range: i64,
    rng: &mut StdRng,
) -> Result<TensorData, vtc_ir::IrError> {
    let numel = shape.numel()?;
    let data = match numeric_dtype(dtype) {
        DType::F32 => TensorData::F32(
            (0..numel)
                .map(|_| bounded_dyadic_f32(value_range, rng))
                .collect(),
        ),
        DType::F64 => TensorData::F64(
            (0..numel)
                .map(|_| bounded_dyadic(value_range, rng))
                .collect(),
        ),
        DType::I32 => TensorData::I32(
            (0..numel)
                .map(|_| clamp_i32(bounded_i64(value_range, rng)))
                .collect(),
        ),
        DType::I64 | DType::Bool => {
            TensorData::I64((0..numel).map(|_| bounded_i64(value_range, rng)).collect())
        }
    };
    Ok(data)
}

fn random_shape(cfg: &GenConfig, rng: &mut StdRng) -> Shape {
    let rank = rng.gen_range(1..=MAX_RANK);
    Shape::new((0..rank).map(|_| Dim::new(random_dim(cfg, rng))).collect())
}

fn random_matrix_shape(cfg: &GenConfig, rng: &mut StdRng) -> Shape {
    Shape::new(vec![
        Dim::new(random_dim(cfg, rng)),
        Dim::new(random_dim(cfg, rng)),
    ])
}

fn random_dim(cfg: &GenConfig, rng: &mut StdRng) -> usize {
    rng.gen_range(1..=cfg.max_dim.max(1))
}

fn random_axes(rank: usize, rng: &mut StdRng) -> Vec<usize> {
    (0..rank).filter(|_| rng.gen_bool(0.5)).collect()
}

fn random_reshape_shape(shape: &Shape, rng: &mut StdRng) -> Shape {
    let numel = shape.numel().unwrap_or(1).max(1);
    let mut candidates = vec![
        Shape::new(vec![Dim::new(numel)]),
        Shape::new(vec![Dim::new(1), Dim::new(numel)]),
        Shape::new(vec![Dim::new(numel), Dim::new(1)]),
    ];
    for divisor in 1..=numel {
        if numel.is_multiple_of(divisor) {
            candidates.push(Shape::new(vec![
                Dim::new(divisor),
                Dim::new(numel / divisor),
            ]));
        }
    }
    candidates[random_index(candidates.len(), rng)].clone()
}

fn sum_output_shape(input_shape: &Shape, axes: &[usize], keepdim: bool) -> Shape {
    if keepdim {
        Shape::new(
            input_shape
                .dims()
                .iter()
                .copied()
                .enumerate()
                .map(|(axis, dim)| {
                    if axes.contains(&axis) {
                        Dim::new(1)
                    } else {
                        dim
                    }
                })
                .collect(),
        )
    } else {
        Shape::new(
            input_shape
                .dims()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim))
                .collect(),
        )
    }
}

fn random_index(len: usize, rng: &mut StdRng) -> usize {
    rng.gen_range(0..len)
}

fn numeric_dtype(dtype: DType) -> DType {
    if dtype.is_numeric() {
        dtype
    } else {
        DType::I64
    }
}

fn bounded_i64(value_range: i64, rng: &mut StdRng) -> i64 {
    let limit = value_range.saturating_abs();
    rng.gen_range(-limit..=limit)
}

fn bounded_dyadic(value_range: i64, rng: &mut StdRng) -> f64 {
    clamped_f64(bounded_i64(value_range, rng)) / 2.0
}

fn bounded_dyadic_f32(value_range: i64, rng: &mut StdRng) -> f32 {
    clamped_f32(bounded_i64(value_range, rng)) / 2.0
}

fn clamped_f32(value: i64) -> f32 {
    f32::from(clamp_i16(value))
}

fn clamped_f64(value: i64) -> f64 {
    f64::from(clamp_i32(value))
}

fn clamp_i16(value: i64) -> i16 {
    match i16::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => i16::MIN,
        Err(_) => i16::MAX,
    }
}

fn clamp_i32(value: i64) -> i32 {
    match i32::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => i32::MIN,
        Err(_) => i32::MAX,
    }
}
