//! Integration tests for oracle-gated autotuning.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};
use vtc_loopir::{
    AffineExpr, Kernel, ScalarExpr, Stmt, TensorF64, eval_loops, eval_loops_f64, lower,
};
use vtc_rewrite::r#gen::{GenConfig, random_graph, random_inputs};
use vtc_schedule::{
    CostModel, Mode, MoveDesc, StaticCost, TuneConfig, autotune, legal_moves, validate_equiv,
};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn input(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> NodeId {
    builder
        .input(name, DType::I64, shape(dims))
        .expect("input is valid")
}

fn const_i64(builder: &mut GraphBuilder, dims: &[usize], values: &[i64]) -> NodeId {
    builder
        .constant(TensorData::I64(values.to_vec()), shape(dims))
        .expect("constant is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn add_relu_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let left = input(&mut builder, "left", &[2, 3]);
    let right = input(&mut builder, "right", &[2, 3]);
    let add = builder.add(left, right).expect("add is valid");
    let relu = builder.relu(add).expect("relu is valid");
    finish(builder, relu)
}

fn matmul_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let lhs = const_i64(
        &mut builder,
        &[4, 4],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let rhs = const_i64(
        &mut builder,
        &[4, 4],
        &[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    );
    let matmul = builder.matmul(lhs, rhs).expect("matmul is valid");
    finish(builder, matmul)
}

fn f64_input_map(graph: &Graph) -> HashMap<String, TensorF64> {
    let mut out = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).expect("input node exists");
        let vtc_ir::Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("shape numel fits");
        let values = (0..numel)
            .map(|index| {
                if index.is_multiple_of(5) {
                    -0.0
                } else {
                    f64::from(u32::try_from(index + 1).expect("index fits u32")) / 2.0
                }
            })
            .collect::<Vec<_>>();
        let tensor = TensorF64::from_f64(ty.shape().clone(), &values).expect("tensor is valid");
        out.insert(name.clone(), tensor);
    }
    out
}

fn cfg(mode: Mode) -> TuneConfig {
    TuneConfig {
        mode,
        tile_sizes: vec![2, 4],
        max_rounds: 4,
        validation_trials: 8,
        restarts: 0,
        seed: 0x1500,
    }
}

#[test]
fn autotune_random_graphs_remain_oracle_equivalent() {
    let graph_cfg = GenConfig {
        seed: 0x1501,
        size: 10,
        max_dim: 4,
        ..GenConfig::default()
    };
    let cost = StaticCost;

    for iteration in 0..24 {
        let mut rng = StdRng::seed_from_u64(graph_cfg.seed.wrapping_add(iteration));
        let (graph, _) = random_graph(&graph_cfg, &mut rng);
        let inputs = random_inputs(&graph, graph_cfg.value_range, &mut rng);
        let kernel = lower(&graph).expect("graph lowers");

        for mode in [Mode::Strict, Mode::FastMath] {
            let tune_cfg = cfg(mode);
            let result = autotune(&kernel, &cost, &tune_cfg).expect("autotune succeeds");
            assert!(result.cost_after <= result.cost_before);
            assert_eq!(
                eval_loops(&kernel, &inputs).expect("original evaluates"),
                eval_loops(&result.kernel, &inputs).expect("tuned evaluates"),
            );

            if mode == Mode::Strict {
                let f64_inputs = f64_input_map(&graph);
                let original = eval_loops_f64(&kernel, &f64_inputs).expect("original f64 eval");
                let tuned = eval_loops_f64(&result.kernel, &f64_inputs).expect("tuned f64 eval");
                assert_eq!(original.len(), tuned.len());
                assert!(
                    original
                        .iter()
                        .zip(&tuned)
                        .all(|(left, right)| left.bit_eq(right))
                );
            }
        }
    }
}

#[test]
fn autotune_finds_fusion_for_add_relu() {
    let graph = add_relu_graph();
    let kernel = lower(&graph).expect("graph lowers");
    let tune_cfg = cfg(Mode::Strict);
    let result = autotune(&kernel, &StaticCost, &tune_cfg).expect("autotune succeeds");

    println!(
        "add+relu moves={:?} cost {} -> {}",
        result.moves, result.cost_before, result.cost_after
    );
    assert!(
        result
            .moves
            .iter()
            .any(|desc| matches!(desc, MoveDesc::Fuse { .. }))
    );
    assert!(result.cost_after < result.cost_before);
    assert!(validate_equiv(
        &kernel,
        &result.kernel,
        Mode::Strict,
        8,
        &mut StdRng::seed_from_u64(0x1502),
    ));
}

#[test]
fn autotune_finds_tile_for_matmul() {
    let kernel = lower(&matmul_graph()).expect("graph lowers");
    let tune_cfg = cfg(Mode::Strict);
    let result = autotune(&kernel, &StaticCost, &tune_cfg).expect("autotune succeeds");

    println!(
        "matmul moves={:?} cost {} -> {}",
        result.moves, result.cost_before, result.cost_after
    );
    assert!(
        result
            .moves
            .iter()
            .any(|desc| matches!(desc, MoveDesc::Tile { .. }))
    );
    assert!(result.cost_after < result.cost_before);
    assert!(validate_equiv(
        &kernel,
        &result.kernel,
        Mode::Strict,
        8,
        &mut StdRng::seed_from_u64(0x1503),
    ));
}

#[test]
fn legal_moves_include_expected_matmul_moves_and_exclude_strict_reduction_swap() {
    let kernel = lower(&matmul_graph()).expect("graph lowers");
    let tune_cfg = cfg(Mode::Strict);
    let moves = legal_moves(&kernel, &tune_cfg)
        .into_iter()
        .map(|(desc, _)| desc)
        .collect::<Vec<_>>();

    assert!(moves.contains(&MoveDesc::Interchange { nest: 1, level: 0 }));
    assert!(moves.contains(&MoveDesc::Tile {
        nest: 1,
        level: 0,
        size: 2,
    }));
    assert!(moves.contains(&MoveDesc::Tile {
        nest: 1,
        level: 2,
        size: 2,
    }));
    assert!(!moves.contains(&MoveDesc::Interchange { nest: 1, level: 1 }));
}

#[test]
fn oracle_gate_rejects_buggy_candidate() {
    let graph = add_relu_graph();
    let kernel = lower(&graph).expect("graph lowers");
    let buggy = corrupt_first_store(&kernel);
    let cost = StaticCost;
    let tune_cfg = cfg(Mode::Strict);

    assert!(!validate_equiv(
        &kernel,
        &buggy,
        Mode::Strict,
        8,
        &mut StdRng::seed_from_u64(0x1504),
    ));

    let result = autotune_with_buggy_candidate(&kernel, &buggy, &cost, &tune_cfg);
    println!(
        "buggy candidate rejected={}",
        result.to_text() != buggy.to_text()
    );
    assert_ne!(result.to_text(), buggy.to_text());
    assert!(validate_equiv(
        &kernel,
        &result,
        Mode::Strict,
        8,
        &mut StdRng::seed_from_u64(0x1505),
    ));
}

#[test]
fn autotune_is_deterministic() {
    let kernel = lower(&add_relu_graph()).expect("graph lowers");
    let tune_cfg = cfg(Mode::Strict);
    let first = autotune(&kernel, &StaticCost, &tune_cfg).expect("autotune succeeds");
    let second = autotune(&kernel, &StaticCost, &tune_cfg).expect("autotune succeeds");

    assert_eq!(first.moves, second.moves);
    assert_eq!(first.cost_before, second.cost_before);
    assert_eq!(first.cost_after, second.cost_after);
    assert_eq!(first.kernel.to_text(), second.kernel.to_text());
}

fn corrupt_first_store(kernel: &Kernel) -> Kernel {
    let mut body = kernel.body().to_vec();
    assert!(corrupt_stmt(&mut body[0]));
    Kernel::new_with_output_shapes(
        kernel.buffers().to_vec(),
        body,
        kernel.inputs().to_vec(),
        kernel.outputs().to_vec(),
        kernel.output_shapes().to_vec(),
    )
}

fn corrupt_stmt(stmt: &mut Stmt) -> bool {
    match stmt {
        Stmt::For { body, .. } => body.iter_mut().any(corrupt_stmt),
        Stmt::Assign { target, value } => {
            target.index = AffineExpr::constant(0);
            *value = ScalarExpr::Sub(Box::new(value.clone()), Box::new(value.clone()));
            true
        }
    }
}

fn autotune_with_buggy_candidate(
    kernel: &Kernel,
    buggy: &Kernel,
    cost: &dyn CostModel,
    cfg: &TuneConfig,
) -> Kernel {
    let mut current = kernel.clone();
    let mut current_cost = cost.cost(&current);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    for _ in 0..cfg.max_rounds {
        let mut candidates = legal_moves(&current, cfg);
        candidates.push((
            MoveDesc::Tile {
                nest: 0,
                level: 0,
                size: 999,
            },
            buggy.clone(),
        ));
        candidates.sort_by_key(|(desc, candidate)| (cost.cost(candidate), desc.clone()));
        let mut adopted = false;
        for (_, candidate) in candidates {
            let candidate_cost = cost.cost(&candidate);
            if candidate_cost >= current_cost {
                continue;
            }
            if validate_equiv(
                kernel,
                &candidate,
                cfg.mode,
                cfg.validation_trials,
                &mut rng,
            ) {
                current = candidate;
                current_cost = candidate_cost;
                adopted = true;
                break;
            }
        }
        if !adopted {
            break;
        }
    }
    current
}
