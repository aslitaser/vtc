# Lean proofs

This is a standalone Lean 4 project. Cargo does not build it.

The rational tensor model represents `n` flat elements as `Fin n → ℚ`. Its
proofs cover double negation, ReLU idempotence, reshape fusion, and the
commutativity and associativity of rational addition.

The [strict scalar model](VtcProofs/StrictF64.lean) uses Lean's logical
binary64 operations. In namespace `VtcProofs.StrictF64`, `decode_eq_iff`
characterizes equality of decoded `UInt64` inputs: their raw bits agree or
both encode NaNs. Thus opposite signed zeros remain distinct observations.
`decode_toBits` proves that every model value round-trips through its canonical
encoding.

The [v1 graph decoder](VtcProofs/GraphEncoding.lean) accepts UTF-8 JSON
for inputs, finite constants, elementwise add, and ReLU. The
[graph evaluator](VtcProofs/Graph.lean) uses the strict scalar operations.
See the [wire format and command-line example](graph-v1.md) and the
[strict f64 contract](strict-f64.md) for the verification boundary.

Build and audit with the pinned Lean and mathlib v4.36.0 toolchain:

```sh
cd proofs
lake exe cache get Mathlib.Algebra.Order.Ring.Rat
lake build
lake env lean CheckAxioms.lean
lake env lean --run GraphEncodingTests.lean
```

CI audits every declaration defined in the proof library, including its
transitive dependencies on axioms from imported modules. The allowlist is
`propext`, `Classical.choice`, and `Quot.sound`. A proof using `sorry`,
`admit`, a native evaluation axiom, or any other unapproved axiom fails the
audit. Import new proof modules from `VtcProofs.lean` (directly or
transitively) so they are built and audited. The audit fails on missing imports.
