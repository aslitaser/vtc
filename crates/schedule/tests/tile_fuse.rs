//! Integration tests for strip-mining and adjacent fusion.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};
use vtc_loopir::{
    AffineExpr, Buffer, BufferId, BufferRef, BufferRole, Kernel, ScalarExpr, Stmt, eval_loops,
    lower,
};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};
use vtc_schedule::{LegalityError, Mode, fuse, interchange, tile};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn const_i64(builder: &mut GraphBuilder, dims: &[usize], values: &[i64]) -> NodeId {
    builder
        .constant(TensorData::I64(values.to_vec()), shape(dims))
        .expect("constant is valid")
}

fn input(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> NodeId {
    builder
        .input(name, DType::I64, shape(dims))
        .expect("input is valid")
}

fn finish(builder: GraphBuilder, outputs: &[NodeId]) -> Graph {
    let mut builder = builder;
    for &output in outputs {
        builder.mark_output(output).expect("output is valid");
    }
    builder.build().expect("graph is valid")
}

fn matmul_graph(rows: usize, depth: usize, cols: usize) -> Graph {
    let mut builder = GraphBuilder::new();
    let left = (1..=rows * depth)
        .map(|value| i64::try_from(value).expect("test value fits i64"))
        .collect::<Vec<_>>();
    let right = (1..=depth * cols)
        .map(|value| i64::try_from(value).expect("test value fits i64"))
        .collect::<Vec<_>>();
    let lhs = const_i64(&mut builder, &[rows, depth], &left);
    let rhs = const_i64(&mut builder, &[depth, cols], &right);
    let matmul = builder.matmul(lhs, rhs).expect("matmul is valid");
    finish(builder, &[matmul])
}

fn add_relu_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = const_i64(&mut builder, &[2, 3], &[1, -2, 3, -4, 5, -6]);
    let y = const_i64(&mut builder, &[2, 3], &[10, 20, -30, 40, -50, 60]);
    let add = builder.add(x, y).expect("add is valid");
    let relu = builder.relu(add).expect("relu is valid");
    finish(builder, &[relu])
}

fn assert_equivalent(original: &Kernel, transformed: &Kernel) {
    assert_eq!(
        eval_loops(original, &HashMap::new()).expect("original kernel evaluates"),
        eval_loops(transformed, &HashMap::new()).expect("transformed kernel evaluates"),
    );
}

#[test]
fn tiles_matmul_parallel_and_reduction_loops() {
    let kernel = lower(&matmul_graph(4, 2, 4)).expect("lowering succeeds");

    for mode in [Mode::Strict, Mode::FastMath] {
        let tiled_m = tile(&kernel, 1, 0, 2, mode).expect("m tile is legal");
        assert_equivalent(&kernel, &tiled_m);
        if mode == Mode::Strict {
            println!("tiled matmul kernel:\n{}", tiled_m.to_text());
        }

        let tiled_k = tile(&kernel, 1, 2, 2, mode).expect("k tile is legal");
        assert_equivalent(&kernel, &tiled_k);
    }
}

#[test]
fn refuses_non_divisible_tile_size() {
    let kernel = lower(&matmul_graph(4, 2, 4)).expect("lowering succeeds");

    assert!(matches!(
        tile(&kernel, 1, 0, 3, Mode::Strict),
        Err(LegalityError::NonDivisibleTile {
            extent: 4,
            tile_size: 3
        }),
    ));
}

#[test]
fn cache_blocking_composes_tile_and_parallel_interchange() {
    let kernel = lower(&matmul_graph(4, 2, 4)).expect("lowering succeeds");
    let tiled_m = tile(&kernel, 1, 0, 2, Mode::Strict).expect("m tile is legal");
    let tiled_mn = tile(&tiled_m, 1, 2, 2, Mode::Strict).expect("n tile is legal");
    let grouped = interchange(&tiled_mn, 1, 1, Mode::Strict).expect("parallel interchange works");

    assert_equivalent(&kernel, &grouped);
}

#[test]
fn fuses_add_then_relu_in_both_modes() {
    let kernel = lower(&add_relu_graph()).expect("lowering succeeds");

    for mode in [Mode::Strict, Mode::FastMath] {
        let fused = fuse(&kernel, 0, 1, mode).expect("add/relu fusion is legal");
        assert_equivalent(&kernel, &fused);
        if mode == Mode::Strict {
            println!("fused add relu kernel:\n{}", fused.to_text());
        }
    }
}

#[test]
fn refuses_invalid_fusions() {
    assert!(matches!(
        fuse(&three_nest_elementwise_kernel(), 0, 2, Mode::Strict),
        Err(LegalityError::FusionNotAdjacent),
    ));

    assert!(matches!(
        fuse(&mismatched_bounds_kernel(), 0, 1, Mode::Strict),
        Err(LegalityError::FusionBoundsMismatch),
    ));

    assert!(matches!(
        fuse(&misaligned_pair_kernel(), 0, 1, Mode::Strict),
        Err(LegalityError::FusionDependenceViolation),
    ));

    let reduction_consumer = lower(&matmul_then_relu_graph()).expect("lowering succeeds");
    assert!(matches!(
        fuse(&reduction_consumer, 1, 2, Mode::Strict),
        Err(LegalityError::FusionBoundsMismatch),
    ));
}

#[test]
fn rational_backstop_catches_bogus_misaligned_fusion() {
    let kernel = misaligned_pair_kernel();
    let bogus = bogus_misaligned_fusion(&kernel);

    assert_ne!(
        eval_loops(&kernel, &HashMap::new()).expect("original kernel evaluates"),
        eval_loops(&bogus, &HashMap::new()).expect("bogus kernel evaluates"),
    );
}

#[test]
fn random_tiles_and_fusions_preserve_loop_results() {
    let cfg = GenConfig {
        seed: 0x5c13_0001,
        size: 12,
        max_dim: 4,
        ..GenConfig::default()
    };
    let mut checked = 0usize;

    for iteration in 0..48 {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        let kernel = lower(&graph).expect("random graph lowers");
        let original = eval_loops(&kernel, &inputs).expect("original evaluates");

        if let Some((nest, level, tile_size)) = first_divisible_tile(&kernel) {
            for mode in [Mode::Strict, Mode::FastMath] {
                let tiled = tile(&kernel, nest, level, tile_size, mode).expect("tile is legal");
                assert_eq!(
                    original,
                    eval_loops(&tiled, &inputs).expect("tiled kernel evaluates"),
                );
                checked = checked.saturating_add(1);
            }
        }

        for nest in 0..kernel.body().len().saturating_sub(1) {
            for mode in [Mode::Strict, Mode::FastMath] {
                if let Ok(fused) = fuse(&kernel, nest, nest.saturating_add(1), mode) {
                    assert_eq!(
                        original,
                        eval_loops(&fused, &inputs).expect("fused kernel evaluates"),
                    );
                    checked = checked.saturating_add(1);
                }
            }
        }
    }

    assert!(
        checked > 0,
        "random test should exercise at least one transform"
    );
}

fn first_divisible_tile(kernel: &Kernel) -> Option<(usize, usize, usize)> {
    for (nest_index, stmt) in kernel.body().iter().enumerate() {
        let levels = perfect_levels(stmt)?;
        for (level_index, (_, lo, hi)) in levels.iter().enumerate() {
            if *lo == 0 && *hi > 1 && hi % 2 == 0 {
                return Some((nest_index, level_index, 2));
            }
        }
    }
    None
}

fn perfect_levels(stmt: &Stmt) -> Option<Vec<(vtc_loopir::LoopVar, i64, i64)>> {
    let mut levels = Vec::new();
    let mut current = stmt;
    loop {
        match current {
            Stmt::For { var, lo, hi, body } => {
                let lo = constant_affine(lo)?;
                let hi = constant_affine(hi)?;
                levels.push((*var, lo, hi));
                if body.len() == 1 && matches!(body.first(), Some(Stmt::For { .. })) {
                    current = body.first()?;
                } else if body
                    .iter()
                    .all(|inner| matches!(inner, Stmt::Assign { .. }))
                {
                    return Some(levels);
                } else {
                    return None;
                }
            }
            Stmt::Assign { .. } => return None,
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

fn three_nest_elementwise_kernel() -> Kernel {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", &[2, 2]);
    let y = input(&mut builder, "y", &[2, 2]);
    let add = builder.add(x, y).expect("add is valid");
    let neg = builder.neg(add).expect("neg is valid");
    let relu = builder.relu(neg).expect("relu is valid");
    lower(&finish(builder, &[relu])).expect("lowering succeeds")
}

fn mismatched_bounds_kernel() -> Kernel {
    let mut builder = GraphBuilder::new();
    let x = const_i64(&mut builder, &[2, 2], &[1, 2, 3, 4]);
    let y = const_i64(&mut builder, &[2, 2], &[5, 6, 7, 8]);
    let add = builder.add(x, y).expect("add is valid");
    let z = const_i64(&mut builder, &[4], &[1, 2, 3, 4]);
    let relu = builder.relu(z).expect("relu is valid");
    lower(&finish(builder, &[add, relu])).expect("lowering succeeds")
}

fn matmul_then_relu_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let a = const_i64(&mut builder, &[2, 2], &[1, 2, 3, 4]);
    let b = const_i64(&mut builder, &[2, 2], &[5, 6, 7, 8]);
    let matmul = builder.matmul(a, b).expect("matmul is valid");
    let relu = builder.relu(matmul).expect("relu is valid");
    finish(builder, &[relu])
}

fn misaligned_pair_kernel() -> Kernel {
    let data = Buffer::new(
        BufferId::new(0),
        "data".to_owned(),
        shape(&[2]),
        BufferRole::Const(TensorData::I64(vec![10, 20])),
    );
    let tmp = Buffer::new(
        BufferId::new(1),
        "tmp".to_owned(),
        shape(&[3]),
        BufferRole::Temp,
    );
    let out = Buffer::new(
        BufferId::new(2),
        "out".to_owned(),
        shape(&[2]),
        BufferRole::Temp,
    );
    let i = vtc_loopir::LoopVar::new(0);
    let j = vtc_loopir::LoopVar::new(1);
    Kernel::new(
        vec![data, tmp, out],
        vec![
            Stmt::For {
                var: i,
                lo: AffineExpr::constant(0),
                hi: AffineExpr::constant(2),
                body: vec![Stmt::Assign {
                    target: BufferRef {
                        buffer: BufferId::new(1),
                        index: AffineExpr::var(i),
                    },
                    value: ScalarExpr::Load(BufferRef {
                        buffer: BufferId::new(0),
                        index: AffineExpr::var(i),
                    }),
                }],
            },
            Stmt::For {
                var: j,
                lo: AffineExpr::constant(0),
                hi: AffineExpr::constant(2),
                body: vec![Stmt::Assign {
                    target: BufferRef {
                        buffer: BufferId::new(2),
                        index: AffineExpr::var(j),
                    },
                    value: ScalarExpr::Load(BufferRef {
                        buffer: BufferId::new(1),
                        index: AffineExpr::new(vec![(j, 1)], 1),
                    }),
                }],
            },
        ],
        Vec::new(),
        vec![BufferId::new(2)],
    )
}

fn bogus_misaligned_fusion(kernel: &Kernel) -> Kernel {
    let Stmt::For { body: first, .. } = &kernel.body()[0] else {
        panic!("test kernel has expected first loop");
    };
    let Stmt::For { body: second, .. } = &kernel.body()[1] else {
        panic!("test kernel has expected second loop");
    };
    let i = vtc_loopir::LoopVar::new(0);
    let j = vtc_loopir::LoopVar::new(1);
    let second = second
        .iter()
        .map(|stmt| rename_stmt(stmt, j, i))
        .collect::<Vec<_>>();
    let mut fused_body = first.clone();
    fused_body.extend(second);
    Kernel::new(
        kernel.buffers().to_vec(),
        vec![Stmt::For {
            var: i,
            lo: AffineExpr::constant(0),
            hi: AffineExpr::constant(2),
            body: fused_body,
        }],
        kernel.inputs().to_vec(),
        kernel.outputs().to_vec(),
    )
}

fn rename_stmt(stmt: &Stmt, from: vtc_loopir::LoopVar, to: vtc_loopir::LoopVar) -> Stmt {
    match stmt {
        Stmt::For { var, lo, hi, body } => Stmt::For {
            var: if *var == from { to } else { *var },
            lo: lo.substitute(from, &AffineExpr::var(to)),
            hi: hi.substitute(from, &AffineExpr::var(to)),
            body: body
                .iter()
                .map(|inner| rename_stmt(inner, from, to))
                .collect(),
        },
        Stmt::Assign { target, value } => Stmt::Assign {
            target: BufferRef {
                buffer: target.buffer,
                index: target.index.substitute(from, &AffineExpr::var(to)),
            },
            value: rename_scalar(value, from, to),
        },
    }
}

fn rename_scalar(
    value: &ScalarExpr,
    from: vtc_loopir::LoopVar,
    to: vtc_loopir::LoopVar,
) -> ScalarExpr {
    match value {
        ScalarExpr::Load(reference) => ScalarExpr::Load(BufferRef {
            buffer: reference.buffer,
            index: reference.index.substitute(from, &AffineExpr::var(to)),
        }),
        ScalarExpr::ConstScalar(value) => ScalarExpr::ConstScalar(value.clone()),
        ScalarExpr::Add(left, right) => ScalarExpr::Add(
            Box::new(rename_scalar(left, from, to)),
            Box::new(rename_scalar(right, from, to)),
        ),
        ScalarExpr::Sub(left, right) => ScalarExpr::Sub(
            Box::new(rename_scalar(left, from, to)),
            Box::new(rename_scalar(right, from, to)),
        ),
        ScalarExpr::Mul(left, right) => ScalarExpr::Mul(
            Box::new(rename_scalar(left, from, to)),
            Box::new(rename_scalar(right, from, to)),
        ),
        ScalarExpr::Neg(input) => ScalarExpr::Neg(Box::new(rename_scalar(input, from, to))),
        ScalarExpr::Relu(input) => ScalarExpr::Relu(Box::new(rename_scalar(input, from, to))),
    }
}
