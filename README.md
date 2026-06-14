# vtc

Small tensor compiler prototype in Rust, with a separate Lean proof project for
the first rewrite laws.

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

Build the Lean proofs:

```sh
cd proofs
lake exe cache get
lake build
```
