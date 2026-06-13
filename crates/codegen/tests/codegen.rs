//! Integration tests for C emission, execution, and measured-cost tuning.

use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use vtc_codegen::{
    BenchError, MeasuredCost, compile_and_run, emit_c, has_c_compiler, measure_runtime,
};
use vtc_interp::Tensor;
use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape, TensorData};
use vtc_loopir::{Kernel, TensorF64, eval_loops, eval_loops_f64, lower};
use vtc_schedule::{Mode, StaticCost, TuneConfig, autotune, fuse, tile};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect())
}

fn input_f64(builder: &mut GraphBuilder, name: &str, dims: &[usize]) -> NodeId {
    builder
        .input(name, DType::F64, shape(dims))
        .expect("input is valid")
}

fn const_i64(builder: &mut GraphBuilder, dims: &[usize], values: &[i64]) -> NodeId {
    builder
        .constant(TensorData::I64(values.to_vec()), shape(dims))
        .expect("constant is valid")
}

fn const_f64(builder: &mut GraphBuilder, dims: &[usize], values: &[f64]) -> NodeId {
    builder
        .constant(TensorData::F64(values.to_vec()), shape(dims))
        .expect("constant is valid")
}

fn finish(builder: GraphBuilder, output: NodeId) -> Graph {
    let mut builder = builder;
    builder.mark_output(output).expect("output is valid");
    builder.build().expect("graph is valid")
}

fn matmul_graph(size: usize) -> Graph {
    let mut builder = GraphBuilder::new();
    let count = size.checked_mul(size).expect("test size fits");
    let lhs_values = (1..=i64::try_from(count).expect("test count fits")).collect::<Vec<_>>();
    let rhs_values = (1..=i64::try_from(count).expect("test count fits"))
        .rev()
        .collect::<Vec<_>>();
    let lhs = const_i64(&mut builder, &[size, size], &lhs_values);
    let rhs = const_i64(&mut builder, &[size, size], &rhs_values);
    let matmul = builder.matmul(lhs, rhs).expect("matmul is valid");
    finish(builder, matmul)
}

fn sum_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let input = const_f64(&mut builder, &[3], &[1e16, 1.0, -1e16]);
    let sum = builder.sum(input, vec![0], false).expect("sum is valid");
    finish(builder, sum)
}

fn add_relu_graph() -> Graph {
    let mut builder = GraphBuilder::new();
    let left = input_f64(&mut builder, "left", &[2, 3]);
    let right = input_f64(&mut builder, "right", &[2, 3]);
    let add = builder.add(left, right).expect("add is valid");
    let relu = builder.relu(add).expect("relu is valid");
    finish(builder, relu)
}

fn f64_inputs(graph: &Graph) -> HashMap<String, TensorF64> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).expect("input node exists");
        let vtc_ir::Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("shape numel fits");
        let values = (0..numel)
            .map(|index| match index {
                0 => -0.0,
                1 => 0.0,
                _ => f64::from(u32::try_from(index).expect("index fits")) - 2.5,
            })
            .collect::<Vec<_>>();
        let tensor = TensorF64::from_f64(ty.shape().clone(), &values).expect("tensor is valid");
        inputs.insert(name.clone(), tensor);
    }
    inputs
}

fn rational_inputs(graph: &Graph) -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    for &input_id in graph.inputs() {
        let node = graph.node(input_id).expect("input node exists");
        let vtc_ir::Op::Input { name, ty } = node.op() else {
            continue;
        };
        let numel = ty.shape().numel().expect("shape numel fits");
        let values = (0..numel)
            .map(|index| i64::try_from(index).expect("index fits") - 2)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_i64(ty.shape().clone(), &values).expect("tensor is valid");
        inputs.insert(name.clone(), tensor);
    }
    inputs
}

fn assert_bit_eq(left: &[TensorF64], right: &[TensorF64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!(left.bit_eq(right), "left={left:?}, right={right:?}");
    }
}

fn assert_c_matches_loop(kernel: &Kernel, inputs: &HashMap<String, TensorF64>) {
    if !has_c_compiler() {
        eprintln!("skipping C execution check: no C compiler found");
        return;
    }
    let c_outputs = compile_and_run(kernel, inputs).expect("compiled C kernel runs");
    let loop_outputs = eval_loops_f64(kernel, inputs).expect("loop f64 oracle runs");
    assert_bit_eq(&c_outputs, &loop_outputs);
}

#[test]
fn emitted_c_is_deterministic_and_contains_concrete_contract() {
    let kernel = lower(&matmul_graph(2)).expect("graph lowers");
    let first = emit_c(&kernel).expect("C emission succeeds");
    let second = emit_c(&kernel).expect("C emission succeeds");

    assert_eq!(first, second);
    assert!(first.contains("#pragma STDC FP_CONTRACT OFF"));
    assert!(first.contains("static void vtc_kernel("));
    println!("{first}");
}

#[test]
fn compiled_c_matches_loop_f64_for_lowered_and_scheduled_kernels() {
    let matmul = lower(&matmul_graph(2)).expect("matmul lowers");
    assert_c_matches_loop(&matmul, &HashMap::new());

    let sum = lower(&sum_graph()).expect("sum lowers");
    assert_c_matches_loop(&sum, &HashMap::new());

    let add_relu = add_relu_graph();
    let add_relu_inputs = f64_inputs(&add_relu);
    let add_relu_kernel = lower(&add_relu).expect("add+relu lowers");
    assert_c_matches_loop(&add_relu_kernel, &add_relu_inputs);

    let fused = fuse(&add_relu_kernel, 0, 1, Mode::Strict).expect("fusion is legal");
    assert_c_matches_loop(&fused, &add_relu_inputs);

    let tiled = tile(&matmul, 1, 0, 2, Mode::Strict).expect("tiling is legal");
    assert_c_matches_loop(&tiled, &HashMap::new());

    let tuned = autotune(
        &add_relu_kernel,
        &StaticCost,
        &TuneConfig {
            mode: Mode::Strict,
            tile_sizes: vec![2],
            max_rounds: 2,
            validation_trials: 4,
            restarts: 0,
            seed: 0x1600,
        },
    )
    .expect("autotune succeeds");
    assert_c_matches_loop(&tuned.kernel, &add_relu_inputs);
}

#[test]
fn missing_compiler_returns_a_fail_closed_error() {
    let _guard = env_lock().lock().expect("env lock not poisoned");
    let old_cc = env::var_os("CC");
    set_cc(Some(OsString::from("definitely-not-a-vtc-c-compiler")));

    let kernel = lower(&matmul_graph(1)).expect("graph lowers");
    let result = compile_and_run(&kernel, &HashMap::new());

    set_cc(old_cc);
    assert!(matches!(result, Err(BenchError::NoCompiler)));
}

#[test]
fn measured_runtime_returns_a_cost_when_a_compiler_exists() {
    if !has_c_compiler() {
        eprintln!("skipping measured runtime check: no C compiler found");
        return;
    }

    let kernel = lower(&matmul_graph(2)).expect("graph lowers");
    let duration = measure_runtime(&kernel, &HashMap::new(), 2).expect("runtime measures");
    assert!(!duration.is_zero());
}

#[test]
fn measured_cost_autotune_preserves_loop_oracles() {
    if !has_c_compiler() {
        eprintln!("skipping measured-cost autotune check: no C compiler found");
        return;
    }

    let graph = add_relu_graph();
    let kernel = lower(&graph).expect("graph lowers");
    let f64_inputs = f64_inputs(&graph);
    let rational_inputs = rational_inputs(&graph);
    let cost = MeasuredCost {
        inputs: f64_inputs.clone(),
        reps: 2,
    };
    let result = autotune(
        &kernel,
        &cost,
        &TuneConfig {
            mode: Mode::Strict,
            tile_sizes: vec![2],
            max_rounds: 2,
            validation_trials: 4,
            restarts: 0,
            seed: 0x1601,
        },
    )
    .expect("autotune succeeds");

    assert!(result.candidates_evaluated > 0);
    assert_eq!(
        eval_loops(&kernel, &rational_inputs).expect("original rational"),
        eval_loops(&result.kernel, &rational_inputs).expect("tuned rational"),
    );
    assert_bit_eq(
        &eval_loops_f64(&kernel, &f64_inputs).expect("original f64"),
        &eval_loops_f64(&result.kernel, &f64_inputs).expect("tuned f64"),
    );
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn set_cc(value: Option<OsString>) {
    // SAFETY: this test serializes all local CC mutation through `env_lock`.
    // No library code stores borrowed environment pointers across the mutation.
    unsafe {
        if let Some(value) = value {
            env::set_var("CC", value);
        } else {
            env::remove_var("CC");
        }
    }
}
