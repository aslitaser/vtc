//! Integration tests for the capstone CLI pipeline.

use std::collections::HashMap;
use std::process::Command;

use vtc::{
    CheckStatus, CostKind, PipelineError, RunOpts, build_example, list_examples, run_pipeline,
};
use vtc_codegen::has_c_compiler;
use vtc_ir::infer_types;
use vtc_schedule::Mode;

fn opts(example: &str) -> RunOpts {
    RunOpts {
        example: example.to_owned(),
        shapes: HashMap::new(),
        mode: Mode::Strict,
        egg: false,
        cost: CostKind::Static,
        reps: 1,
        seed: 0x1800,
        check: false,
        emit: None,
        no_compile: true,
        verbose: false,
    }
}

#[test]
fn list_examples_are_buildable_and_typed() {
    let examples = list_examples();
    assert!(!examples.is_empty());

    for (name, _) in examples {
        let (graph, inputs) = build_example(name, &HashMap::new()).expect("example builds");
        assert!(!inputs.is_empty());
        infer_types(&graph).expect("example type-checks");
    }
}

#[test]
fn matmul_pipeline_reaches_c_emission_without_compiler() {
    let report = run_pipeline(&opts("matmul")).expect("pipeline succeeds");

    assert!(!report.compiled);
    assert!(report.emitted_c.contains("#pragma STDC FP_CONTRACT OFF"));
    assert_eq!(report.example, "matmul");
}

#[test]
fn add_relu_self_check_passes_and_compiles_when_available() {
    let mut run_opts = opts("add-relu");
    run_opts.check = true;
    run_opts.no_compile = false;

    let report = run_pipeline(&run_opts).expect("pipeline succeeds");

    assert_eq!(report.compiled, has_c_compiler());
    let CheckStatus::Passed(checks) = report.check else {
        panic!("self-check should pass");
    };
    assert!(checks.iter().any(|check| check.contains("rational")));
    if has_c_compiler() {
        assert!(checks.iter().any(|check| check.contains("compiled C")));
    }
}

#[test]
fn modes_and_egg_combinations_stay_rational_equivalent() {
    for mode in [Mode::Strict, Mode::FastMath] {
        for egg in [false, true] {
            let mut run_opts = opts("matmul-relu-sum");
            run_opts.mode = mode;
            run_opts.egg = egg;
            run_opts.check = true;

            let report = run_pipeline(&run_opts).expect("pipeline succeeds");
            assert!(matches!(report.check, CheckStatus::Passed(_)));
        }
    }
}

#[test]
fn same_seed_is_deterministic() {
    let mut run_opts = opts("add-relu");
    run_opts.check = true;
    run_opts.shapes.insert("D0".to_owned(), 4);
    run_opts.shapes.insert("D1".to_owned(), 4);

    let first = run_pipeline(&run_opts).expect("first run succeeds");
    let second = run_pipeline(&run_opts).expect("second run succeeds");

    assert_eq!(first, second);
    assert_eq!(first.emitted_c, second.emitted_c);
}

#[test]
fn invalid_shape_and_unknown_example_fail_closed() {
    let mut bad_shape = HashMap::new();
    bad_shape.insert("M".to_owned(), 0);
    assert!(matches!(
        build_example("matmul", &bad_shape),
        Err(PipelineError::InvalidShape { .. })
    ));

    assert!(matches!(
        build_example("missing", &HashMap::new()),
        Err(PipelineError::UnknownExample(_))
    ));
}

#[test]
fn binary_list_smoke_test() {
    let output = Command::new(env!("CARGO_BIN_EXE_vtc"))
        .arg("list")
        .output()
        .expect("vtc binary runs");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert!(stdout.contains("matmul"));
}
