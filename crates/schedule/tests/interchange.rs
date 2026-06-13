//! Integration tests for dependence classification and loop interchange.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};
use vtc_loopir::{AffineExpr, Kernel, Stmt, eval_loops, lower};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};
use vtc_schedule::{LegalityError, LevelDep, Mode, classify_levels, interchange};

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

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn matmul_kernel() -> Kernel {
    let mut builder = GraphBuilder::new();
    let a = const_i64(&mut builder, &[2, 2], &[1, 2, 3, 4]);
    let b = const_i64(&mut builder, &[2, 2], &[5, 6, 7, 8]);
    let matmul = builder.matmul(a, b).expect("matmul is valid");
    lower(&finish(builder, matmul)).expect("lowering succeeds")
}

fn add_kernel() -> Kernel {
    let mut builder = GraphBuilder::new();
    let a = const_i64(&mut builder, &[2, 3], &[1, 2, 3, 4, 5, 6]);
    let b = const_i64(&mut builder, &[2, 3], &[10, 20, 30, 40, 50, 60]);
    let add = builder.add(a, b).expect("add is valid");
    lower(&finish(builder, add)).expect("lowering succeeds")
}

fn assert_loop_equivalent(original: &Kernel, transformed: &Kernel) {
    assert_eq!(
        eval_loops(original, &HashMap::new()).expect("original kernel evaluates"),
        eval_loops(transformed, &HashMap::new()).expect("transformed kernel evaluates"),
    );
}

#[test]
fn classify_matmul_and_elementwise_levels() {
    let matmul = matmul_kernel();
    assert_eq!(
        classify_levels(&matmul.body()[1]).expect("matmul accumulate nest is perfect"),
        vec![LevelDep::Parallel, LevelDep::Parallel, LevelDep::Reduction],
    );

    let add = add_kernel();
    assert_eq!(
        classify_levels(&add.body()[0]).expect("elementwise nest is perfect"),
        vec![LevelDep::Parallel, LevelDep::Parallel],
    );
}

#[test]
fn swaps_parallel_matmul_levels_in_both_modes() {
    let kernel = matmul_kernel();

    for mode in [Mode::Strict, Mode::FastMath] {
        let swapped = interchange(&kernel, 1, 0, mode).expect("m/n swap is legal");
        assert_loop_equivalent(&kernel, &swapped);
    }
}

#[test]
fn strict_refuses_reduction_reorder_but_fast_math_allows_it() {
    let kernel = matmul_kernel();

    assert!(matches!(
        interchange(&kernel, 1, 1, Mode::Strict),
        Err(LegalityError::ReductionReorderUnderStrict),
    ));

    let fast = interchange(&kernel, 1, 1, Mode::FastMath).expect("n/k swap is fast-math legal");
    assert_loop_equivalent(&kernel, &fast);
}

#[test]
fn rational_backstop_catches_bogus_corruption() {
    let kernel = add_kernel();
    let bad = bogus_swap_and_corrupt(&kernel, 0, 0);

    assert_ne!(
        eval_loops(&kernel, &HashMap::new()).expect("original kernel evaluates"),
        eval_loops(&bad, &HashMap::new()).expect("bogus kernel evaluates"),
    );
}

#[test]
fn random_parallel_interchanges_preserve_loop_results() {
    let cfg = GenConfig {
        seed: 0x5c12_0001,
        size: 12,
        max_dim: 3,
        ..GenConfig::default()
    };
    let mut checked = 0usize;

    for iteration in 0..48 {
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(iteration));
        let (graph, _) = random_graph(&cfg, &mut rng);
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        let kernel = lower(&graph).expect("random graph lowers");
        let original = eval_loops(&kernel, &inputs).expect("original kernel evaluates");

        for (nest_index, stmt) in kernel.body().iter().enumerate() {
            let Ok(levels) = classify_levels(stmt) else {
                continue;
            };
            for level in 0..levels.len().saturating_sub(1) {
                if levels[level] == LevelDep::Parallel
                    && levels[level.saturating_add(1)] == LevelDep::Parallel
                {
                    for mode in [Mode::Strict, Mode::FastMath] {
                        let swapped = interchange(&kernel, nest_index, level, mode)
                            .expect("parallel swap is legal");
                        assert_eq!(
                            original,
                            eval_loops(&swapped, &inputs).expect("swapped kernel evaluates"),
                        );
                        checked = checked.saturating_add(1);
                    }
                }
            }
        }
    }

    assert!(checked > 0, "random test should exercise legal swaps");
}

fn bogus_swap_and_corrupt(kernel: &Kernel, nest: usize, level: usize) -> Kernel {
    let swapped = interchange(kernel, nest, level, Mode::Strict).expect("parallel swap is legal");
    let mut body = swapped.body().to_vec();
    assert!(corrupt_first_assignment(&mut body[nest]));
    Kernel::new_with_output_shapes(
        swapped.buffers().to_vec(),
        body,
        swapped.inputs().to_vec(),
        swapped.outputs().to_vec(),
        swapped.output_shapes().to_vec(),
    )
}

fn corrupt_first_assignment(stmt: &mut Stmt) -> bool {
    match stmt {
        Stmt::For { body, .. } => body.iter_mut().any(corrupt_first_assignment),
        Stmt::Assign { target, .. } => {
            target.index = AffineExpr::constant(0);
            true
        }
    }
}

#[test]
fn input_backed_parallel_swap_preserves_random_inputs() {
    let mut builder = GraphBuilder::new();
    let x = input(&mut builder, "x", &[2, 3]);
    let y = input(&mut builder, "y", &[2, 3]);
    let add = builder.add(x, y).expect("add is valid");
    let graph = finish(builder, add);
    let kernel = lower(&graph).expect("lowering succeeds");
    let swapped = interchange(&kernel, 0, 0, Mode::Strict).expect("parallel swap is legal");

    let cfg = GenConfig::default();
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(0x5c12_1000 + seed);
        let inputs = random_inputs(&graph, cfg.value_range, &mut rng);
        assert_eq!(
            eval_loops(&kernel, &inputs).expect("original kernel evaluates"),
            eval_loops(&swapped, &inputs).expect("swapped kernel evaluates"),
        );
    }
}
