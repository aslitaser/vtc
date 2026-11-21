# vtc

Small tensor compiler prototype in Rust, with a separate Lean project for
rational rewrite laws and strict binary64 scalar semantics.

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Run the built-in examples:

```sh
cargo run -p vtc -- list
cargo run -p vtc -- run --example matmul --check
```

Build and audit the Lean proofs. The proposed strict binary64 semantics and
verification boundary are documented in the [strict f64 contract](proofs/strict-f64.md).

```sh
cd proofs
lake exe cache get Mathlib.Algebra.Order.Ring.Rat
lake build
lake env lean CheckAxioms.lean
```
