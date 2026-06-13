# vtc

`vtc` is the workspace skeleton for a small verified tensor compiler: a graph layer with rational denotational semantics and semantics-preserving rewrites, plus a schedule layer validated per compilation against an affine loop IR. The design contract for later work is in [DESIGN.md](DESIGN.md).

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
