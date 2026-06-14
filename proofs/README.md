# Lean proofs

This is a standalone Lean 4 project. Cargo does not build it.

The model is deliberately small: a tensor with `n` flat elements is
`Fin n -> Q`. The proved rewrites are double negation, relu idempotence, and
reshape fusion. There are also two scalar law lemmas for addition commutativity
and associativity.

These proofs cover rational identities, not the Rust graph surgery code and not
IEEE bit behavior. Rust tests cover the implementation path; the f64 oracle
checks bit behavior.

Build:

```sh
cd proofs
lake exe cache get
lake build
```
