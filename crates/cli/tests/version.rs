//! Integration tests for the `vtc` command-line interface.

use std::process::Command;

#[test]
fn version_flag_succeeds() {
    let output = Command::new(env!("CARGO_BIN_EXE_vtc"))
        .arg("--version")
        .output()
        .expect("vtc binary runs");

    assert!(output.status.success());
}
