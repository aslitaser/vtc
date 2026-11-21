# Strict f64 source graph format, version 1

A graph file contains one UTF-8 JSON array with four fields, in order:

```json
["vtc.graph",1,[["input","f64","x",[2]],["const","f64",[2],["8000000000000000","bff0000000000000"]],["add",0,1],["relu",2]],[3,2,3]]
```

The first two fields identify the format and version. The third is the node
table. A node's ID is its zero-based position in the table, and every operand
must refer to an earlier node. The last field is a nonempty list of output
IDs. The example returns the rectified sum, the original sum, and the
rectified sum again. Order and repetitions are significant.

| Node record | Meaning |
| --- | --- |
| `["input","f64",name,shape]` | Declare a uniquely named input. Input order is declaration order. |
| `["const","f64",shape,bits]` | Declare a finite binary64 constant with one bit pattern per element, in row-major order. |
| `["add",lhs,rhs]` | Add corresponding elements. Both operands must have exactly the same shape. |
| `["relu",src]` | Apply strict binary64 ReLU elementwise and preserve the operand shape. |

Shapes are arrays of positive integer extents whose product fits in an
unsigned 64-bit integer. The empty shape `[]` has one element. IDs are
nonnegative integers. Each constant element is exactly 16 lowercase
hexadecimal digits, most significant digit first, with no prefix. For
example, `"8000000000000000"` encodes negative zero. A constant containing
infinity or NaN is invalid. Runtime inputs may contain either.

The Rust API `vtc_ir::encode_graph_v1(&graph)` exports the original graph.
To try the encoder and independent Lean checker from the repository root:

```sh
cargo run -q -p vtc -- export-graph --example add-relu --shape D0=2,D1=3 > graph.json
(cd proofs && lake build && lake env lean --run CheckGraph.lean ../graph.json)
```

The Lean evaluator takes a flat buffer of [strict scalar values](VtcProofs/StrictF64.lean)
for each named input. `InputsMatch` checks the buffer lengths against the
declared shapes. For every matching input, `decodeBytes_eval_semantics` proves
that an accepted file describes a well-typed graph, that its nodes evaluate
according to their source operations, and that the result selects exactly the
declared outputs in order. `eval_output_typed` identifies the buffer and
logical shape at each output position and proves the buffer has that shape's
element count. [Decoder theorems](VtcProofs/GraphEncoding.lean) · [Evaluation theorems](VtcProofs/Graph.lean)

The proof starts at the input file. The Rust exporter is tested against a
[shared fixture](fixtures/strict-graph-v1.json) that Lean independently decodes
and evaluates. Kernel lowering and C emission are later verification stages.
