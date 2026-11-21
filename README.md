# vtc

Small tensor compiler prototype in Rust, with a separate Lean project for
rational rewrite laws and strict binary64 scalar semantics.

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Run the built-in examples or export the original add–ReLU graph in the
[strict f64 graph format](proofs/graph-v1.md):

```sh
cargo run -p vtc -- list
cargo run -p vtc -- run --example matmul --check
cargo run -p vtc -- export-graph --example add-relu
```

Build and audit the Lean proofs. The proposed strict binary64 semantics and
verification boundary are documented in the [strict f64 contract](proofs/strict-f64.md).

```sh
cd proofs
lake exe cache get Mathlib.Algebra.Order.Ring.Rat
lake build
lake env lean CheckAxioms.lean
```
