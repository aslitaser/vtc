//! Built-in graph examples for the CLI frontend.
//!
//! A real text or JSON graph frontend is future work. These examples are
//! intentionally small and deterministic so the capstone command can exercise
//! the validated pipeline end to end.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape};

use crate::PipelineError;

/// Builds a named example graph and returns it with input names in declaration order.
///
/// Shape parameters are supplied as uppercase names such as `M`, `K`, or `D0`.
/// Missing parameters use small defaults.
///
/// # Errors
///
/// Returns [`PipelineError::UnknownExample`] for an unknown name, or
/// [`PipelineError::InvalidShape`] when a supplied shape parameter is unknown,
/// zero, or otherwise invalid.
pub fn build_example<S: BuildHasher>(
    name: &str,
    shapes: &HashMap<String, usize, S>,
) -> Result<(Graph, Vec<String>), PipelineError> {
    match name {
        "matmul" => build_matmul(shapes),
        "add-relu" => build_add_relu(shapes),
        "matmul-relu-sum" => build_matmul_relu_sum(shapes),
        "two-matmul" => build_two_matmul(shapes),
        other => Err(PipelineError::UnknownExample(other.to_owned())),
    }
}

/// Lists built-in example names with their default shape descriptions.
#[must_use]
pub fn list_examples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("matmul", "M=8,K=8,N=8"),
        ("add-relu", "D0=8,D1=8"),
        ("matmul-relu-sum", "M=8,K=8,N=8"),
        ("two-matmul", "M=4,K=4,H=4,N=4"),
    ]
}

fn build_matmul(
    shapes: &HashMap<String, usize, impl BuildHasher>,
) -> Result<(Graph, Vec<String>), PipelineError> {
    validate_keys(shapes, &["M", "K", "N"])?;
    let m = shape_param(shapes, "M", 8)?;
    let k = shape_param(shapes, "K", 8)?;
    let n = shape_param(shapes, "N", 8)?;

    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "A", &[m, k])?;
    let right = input(&mut builder, "B", &[k, n])?;
    let output = builder.matmul(left, right)?;
    finish(builder, output, &["A", "B"])
}

fn build_add_relu(
    shapes: &HashMap<String, usize, impl BuildHasher>,
) -> Result<(Graph, Vec<String>), PipelineError> {
    validate_keys(shapes, &["D0", "D1"])?;
    let d0 = shape_param(shapes, "D0", 8)?;
    let d1 = shape_param(shapes, "D1", 8)?;

    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "X", &[d0, d1])?;
    let right = input(&mut builder, "Y", &[d0, d1])?;
    let add = builder.add(left, right)?;
    let output = builder.relu(add)?;
    finish(builder, output, &["X", "Y"])
}

fn build_matmul_relu_sum(
    shapes: &HashMap<String, usize, impl BuildHasher>,
) -> Result<(Graph, Vec<String>), PipelineError> {
    validate_keys(shapes, &["M", "K", "N"])?;
    let m = shape_param(shapes, "M", 8)?;
    let k = shape_param(shapes, "K", 8)?;
    let n = shape_param(shapes, "N", 8)?;

    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "A", &[m, k])?;
    let right = input(&mut builder, "B", &[k, n])?;
    let matmul = builder.matmul(left, right)?;
    let relu = builder.relu(matmul)?;
    let output = builder.sum(relu, vec![1], false)?;
    finish(builder, output, &["A", "B"])
}

fn build_two_matmul(
    shapes: &HashMap<String, usize, impl BuildHasher>,
) -> Result<(Graph, Vec<String>), PipelineError> {
    validate_keys(shapes, &["M", "K", "H", "N"])?;
    let m = shape_param(shapes, "M", 4)?;
    let k = shape_param(shapes, "K", 4)?;
    let h = shape_param(shapes, "H", 4)?;
    let n = shape_param(shapes, "N", 4)?;

    let mut builder = GraphBuilder::new();
    let input_x = input(&mut builder, "X", &[m, k])?;
    let weight_1 = input(&mut builder, "W1", &[k, h])?;
    let weight_2 = input(&mut builder, "W2", &[h, n])?;
    let hidden = builder.matmul(input_x, weight_1)?;
    let hidden = builder.relu(hidden)?;
    let output = builder.matmul(hidden, weight_2)?;
    finish(builder, output, &["X", "W1", "W2"])
}

fn input(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> Result<NodeId, PipelineError> {
    builder
        .input(name, DType::F64, shape(dims))
        .map_err(PipelineError::from)
}

fn finish(
    mut builder: GraphBuilder,
    output: NodeId,
    inputs: &[&str],
) -> Result<(Graph, Vec<String>), PipelineError> {
    builder.mark_output(output)?;
    Ok((
        builder.build()?,
        inputs.iter().map(|name| (*name).to_owned()).collect(),
    ))
}

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn shape_param(
    shapes: &HashMap<String, usize, impl BuildHasher>,
    key: &str,
    default: usize,
) -> Result<usize, PipelineError> {
    let value = shapes.get(key).copied().unwrap_or(default);
    if value == 0 {
        return Err(PipelineError::InvalidShape {
            key: key.to_owned(),
            value,
            reason: "dimension must be positive".to_owned(),
        });
    }
    Ok(value)
}

fn validate_keys(
    shapes: &HashMap<String, usize, impl BuildHasher>,
    allowed: &[&str],
) -> Result<(), PipelineError> {
    let allowed = allowed.iter().copied().collect::<HashSet<_>>();
    for key in shapes.keys() {
        if !allowed.contains(key.as_str()) {
            return Err(PipelineError::InvalidShape {
                key: key.clone(),
                value: shapes.get(key).copied().unwrap_or_default(),
                reason: "shape parameter is not used by this example".to_owned(),
            });
        }
    }
    Ok(())
}
