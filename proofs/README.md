# Lean proofs

This is a standalone Lean 4 project. Cargo does not build it.

The model is deliberately small: a tensor with `n` flat elements is
`Fin n -> Q`. The proved rewrites are double negation, relu idempotence, and
reshape fusion. There are also two scalar law lemmas for addition commutativity
and associativity.

These proofs cover rational identities, not the Rust graph surgery code and not
IEEE bit behavior. Rust tests cover the implementation path; the f64 oracle
checks bit behavior.

The [strict f64 contract](strict-f64.md) specifies the target for future
compiler proofs.

Build and audit with the pinned Lean and mathlib v4.36.0 toolchain:

```sh
cd proofs
lake exe cache get Mathlib.Algebra.Order.Ring.Rat
lake build
lake env lean CheckAxioms.lean
```

CI audits every declaration defined in the proof library, including its
transitive dependencies on axioms from imported modules. The allowlist is
`propext`, `Classical.choice`, and `Quot.sound`. A proof using `sorry`,
`admit`, a native evaluation axiom, or any other unapproved axiom fails the
audit. Import new proof modules from `VtcProofs.lean` (directly or
transitively) so they are built and audited. The audit fails on missing imports.
