//! Strict-mode f64 bit-preservation tests for schedule transforms.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Op, Shape, TensorData};
use vtc_loopir::{
    AffineExpr, BufferRef, Kernel, LoopVar, ScalarExpr, Stmt, TensorF64, eval_loops,
    eval_loops_f64, lower,
};
use vtc_schedule::{LevelDep, Mode, classify_levels, fuse, interchange, tile};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn input(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> NodeId {
    builder
        .input(name, DType::F64, shape(dims))
        .expect("input is valid")
}

fn constant_f64(builder: &mut GraphBuilder, dims: &[usize], values: &[f64]) -> NodeId {
    builder
        .constant(TensorData::F64(values.to_vec()), shape(dims))
        .expect("constant is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn add_graph(dims: &[usize]) -> Graph {
    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "left", dims);
    let right = input(&mut builder, "right", dims);
    let add = builder.add(left, right).expect("add is valid");
    finish(builder, add)
}

fn add_relu_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "left", &[2, 3]);
    let right = input(&mut builder, "right", &[2, 3]);
    let add = builder.add(left, right).expect("add is valid");
    let relu = builder.relu(add).expect("relu is valid");
    finish(builder, relu)
}

fn full_sum_graph(values: &[f64], dims: &[usize]) -> Graph {
    let mut builder = GraphBuilder::new();
    let input = constant_f64(&mut builder, dims, values);
    let axes = (0..dims.len()).collect::<Vec<_>>();
    let sum = builder.sum(input, axes, false).expect("sum is valid");
    finish(builder, sum)
}

fn generated_f64_inputs(graph: &Graph, seed: u64) -> HashMap<String, TensorF64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).expect("input id is valid");
        let Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("shape numel fits");
        let values = (0..numel)
            .map(|index| generated_f64(index, &mut rng))
            .collect::<Vec<_>>();
        let tensor = TensorF64::from_f64(ty.shape().clone(), &values).expect("tensor is valid");
        inputs.insert(name.clone(), tensor);
    }
    inputs
}

fn generated_f64(index: usize, rng: &mut StdRng) -> f64 {
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

fn assert_f64_bit_eq(left: &[TensorF64], right: &[TensorF64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!(left.bit_eq(right), "left={left:?}, right={right:?}");
    }
}

fn assert_f64_not_bit_eq(left: &[TensorF64], right: &[TensorF64]) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter()
            .zip(right)
            .any(|(left, right)| !left.bit_eq(right)),
        "expected at least one f64 output to differ",
    );
}

#[test]
fn strict_parallel_interchange_is_f64_bit_identical() {
    let graph = add_graph(&[2, 3]);
    let inputs = generated_f64_inputs(&graph, 0x1400);
    let kernel = lower(&graph).expect("lowering succeeds");
    let swapped = interchange(&kernel, 0, 0, Mode::Strict).expect("parallel swap is legal");

    assert_f64_bit_eq(
        &eval_loops_f64(&kernel, &inputs).expect("original evaluates"),
        &eval_loops_f64(&swapped, &inputs).expect("swapped evaluates"),
    );
}

#[test]
fn strict_tile_is_f64_bit_identical() {
    let graph = add_graph(&[4, 2]);
    let inputs = generated_f64_inputs(&graph, 0x1401);
    let kernel = lower(&graph).expect("lowering succeeds");
    let tiled = tile(&kernel, 0, 0, 2, Mode::Strict).expect("strip-mine is legal");

    assert_f64_bit_eq(
        &eval_loops_f64(&kernel, &inputs).expect("original evaluates"),
        &eval_loops_f64(&tiled, &inputs).expect("tiled evaluates"),
    );
}

#[test]
fn strict_aligned_fusion_is_f64_bit_identical() {
    let graph = add_relu_graph();
    let inputs = generated_f64_inputs(&graph, 0x1402);
    let kernel = lower(&graph).expect("lowering succeeds");
    let fused = fuse(&kernel, 0, 1, Mode::Strict).expect("aligned fusion is legal");

    assert_f64_bit_eq(
        &eval_loops_f64(&kernel, &inputs).expect("original evaluates"),
        &eval_loops_f64(&fused, &inputs).expect("fused evaluates"),
    );
}

#[test]
fn fast_math_reduction_reorder_is_rational_equal_but_can_change_f64_bits() {
    let graph = full_sum_graph(&[1e16, -1e16, 1.0, 0.0], &[2, 2]);
    let kernel = lower(&graph).expect("lowering succeeds");
    let reordered = interchange(&kernel, 1, 0, Mode::FastMath).expect("reduction reorder allowed");

    assert_eq!(
        eval_loops(&kernel, &HashMap::new()).expect("original rational evaluates"),
        eval_loops(&reordered, &HashMap::new()).expect("reordered rational evaluates"),
    );
    assert_f64_not_bit_eq(
        &eval_loops_f64(&kernel, &HashMap::new()).expect("original f64 evaluates"),
        &eval_loops_f64(&reordered, &HashMap::new()).expect("reordered f64 evaluates"),
    );
}

#[test]
fn reverse_reduction_control_is_rational_equal_but_f64_different() {
    let graph = full_sum_graph(&[1e16, -1e16, 1.0], &[3]);
    let kernel = lower(&graph).expect("lowering succeeds");
    let reversed = reverse_first_reduction(&kernel);

    assert_eq!(
        eval_loops(&kernel, &HashMap::new()).expect("original rational evaluates"),
        eval_loops(&reversed, &HashMap::new()).expect("reversed rational evaluates"),
    );
    assert_f64_not_bit_eq(
        &eval_loops_f64(&kernel, &HashMap::new()).expect("original f64 evaluates"),
        &eval_loops_f64(&reversed, &HashMap::new()).expect("reversed f64 evaluates"),
    );
}

fn reverse_first_reduction(kernel: &Kernel) -> Kernel {
    let mut body = kernel.body().to_vec();
    for stmt in &mut body {
        let Ok(levels) = classify_levels(stmt) else {
            continue;
        };
        if let Some((level, _)) = levels
            .iter()
            .enumerate()
            .find(|(_, dep)| **dep == LevelDep::Reduction)
        {
            *stmt = reverse_loop_level(stmt, level);
            return Kernel::new_with_output_shapes(
                kernel.buffers().to_vec(),
                body,
                kernel.inputs().to_vec(),
                kernel.outputs().to_vec(),
                kernel.output_shapes().to_vec(),
            );
        }
    }
    panic!("test kernel must contain a reduction loop");
}

fn reverse_loop_level(stmt: &Stmt, level: usize) -> Stmt {
    match stmt {
        Stmt::For { var, lo, hi, body } if level == 0 => {
            let lo_value = constant_affine(lo).expect("reduction loop has constant lo");
            let hi_value = constant_affine(hi).expect("reduction loop has constant hi");
            assert_eq!(lo_value, 0);
            let replacement = AffineExpr::new(vec![(*var, -1)], hi_value.saturating_sub(1));
            Stmt::For {
                var: *var,
                lo: lo.clone(),
                hi: hi.clone(),
                body: substitute_stmts(body, *var, &replacement),
            }
        }
        Stmt::For { var, lo, hi, body } => Stmt::For {
            var: *var,
            lo: lo.clone(),
            hi: hi.clone(),
            body: body
                .iter()
                .map(|stmt| reverse_loop_level(stmt, level.saturating_sub(1)))
                .collect(),
        },
        Stmt::Assign { .. } => panic!("test kernel must be a perfect nest"),
    }
}

fn substitute_stmts(stmts: &[Stmt], var: LoopVar, replacement: &AffineExpr) -> Vec<Stmt> {
    stmts
        .iter()
        .map(|stmt| substitute_stmt(stmt, var, replacement))
        .collect()
}

fn substitute_stmt(stmt: &Stmt, var: LoopVar, replacement: &AffineExpr) -> Stmt {
    match stmt {
        Stmt::For {
            var: loop_var,
            lo,
            hi,
            body,
        } => Stmt::For {
            var: *loop_var,
            lo: lo.substitute(var, replacement),
            hi: hi.substitute(var, replacement),
            body: substitute_stmts(body, var, replacement),
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: BufferRef {
                buffer: target.buffer,
                index: target.index.substitute(var, replacement),
            },
            value: substitute_scalar(value, var, replacement),
        },
    }
}

fn substitute_scalar(value: &ScalarExpr, var: LoopVar, replacement: &AffineExpr) -> ScalarExpr {
    match value {
        ScalarExpr::Load(reference) => ScalarExpr::Load(BufferRef {
            buffer: reference.buffer,
            index: reference.index.substitute(var, replacement),
        }),
        ScalarExpr::ConstScalar(value) => ScalarExpr::ConstScalar(value.clone()),
        ScalarExpr::Add(left, right) => ScalarExpr::Add(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Sub(left, right) => ScalarExpr::Sub(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Mul(left, right) => ScalarExpr::Mul(
            Box::new(substitute_scalar(left, var, replacement)),
            Box::new(substitute_scalar(right, var, replacement)),
        ),
        ScalarExpr::Neg(input) => {
            ScalarExpr::Neg(Box::new(substitute_scalar(input, var, replacement)))
        }
        ScalarExpr::Relu(input) => {
            ScalarExpr::Relu(Box::new(substitute_scalar(input, var, replacement)))
        }
    }
}

fn constant_affine(expr: &AffineExpr) -> Option<i64> {
    if expr.terms().iter().any(|(_, coeff)| *coeff != 0) {
        None
    } else {
        Some(expr.constant_term())
    }
}
