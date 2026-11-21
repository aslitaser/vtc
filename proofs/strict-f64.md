# Strict f64 contract

This is the specification for vtc's proposed verified compilation path. The
[v1 source graph](graph-v1.md) covers input, constant, add and ReLU. The full subset
also covers subtraction, multiplication, negation, reshape, rank-two matrix
multiplication and sum over explicit axes. Lean defines the
[strict scalar operations and input equality](VtcProofs/StrictF64.lean).
The current `--check` option compares executions on sample inputs.

## Graphs and observed results

Graphs are well typed and acyclic. Inputs, constants and results have type
`F64`. Shapes are fixed, extents are positive, and a rank-zero shape `[]`
contains one element. Storage is row-major. Elementwise operands have equal
shapes. Constants are finite binary64 values, including subnormals and both
zeros. Inputs may have any binary64 encoding, including infinities and NaNs.

Results agree when the ordered outputs have the same shapes and each pair of
corresponding elements either has identical bits or consists of two NaNs.
This distinguishes signed zeros and identifies all NaN encodings, regardless
of sign, payload or signaling status. The observations are output values and
shapes. Floating-point status flags are outside the contract and traps are
disabled. See the [current output comparison](../crates/interp/src/tensor_f64.rs).

## Operations

Use binary64 round-to-nearest, ties-to-even, with gradual underflow. Round
each arithmetic operation separately, including multiplication followed by
addition. The target must preserve this behavior, including for intermediate
values that overflow or underflow.

| Operation | Required result |
| --- | --- |
| Add, subtract, multiply | Apply the corresponding binary64 operation to each pair of elements. |
| Negate | Reverse the sign, including for zero. |
| ReLU | Return the operand when it is strictly greater than zero. Otherwise return `+0`, including for NaN and `-0`. |
| Reshape | Preserve element count and the flat sequence of values. Return the requested logical shape. |
| Matmul | For shapes `(M,K)` and `(K,N)`, return `(M,N)`. Start each dot product at `+0`. In ascending `k` order, add each separately rounded product to the accumulator. |
| Sum | Start each output at `+0`. Visit inputs in flat row-major order and add each to its output bucket. Axes must be distinct and in range. Keep reduced dimensions at extent one when `keepdim` is set, otherwise remove them. |

With an empty list of sum axes, each bucket receives one value. In particular,
an input of `-0` yields `+0` because the accumulator starts at `+0`. This
follows the [graph interpreter](../crates/interp/src/eval_f64.rs).

## Target theorem

The proposed verifier takes the original graph, a certificate and the exact
bytes of a C kernel. If it accepts, then for every admissible input and under
the target and caller preconditions below, the kernel terminates with defined
behavior, preserves its inputs and returns results equivalent to the graph.

- **Target.** Use an LP64 C target with eight-bit bytes, binary64 `double` and
  the floating-point environment above. Compilation must preserve subnormals,
  intermediate rounding and reduction order, including disabling contraction.
- **Caller.** Supply live, correctly aligned input and output buffers of the
  declared extents and sufficient storage for temporaries. Inputs are
  initialized and may share read-only storage. Outputs are writable,
  pairwise disjoint and disjoint from inputs.
- **Verifier.** Establish valid shapes, reshape aliases, initialized reads,
  in-bounds accesses and writes confined to writable buffers. Prove that all
  size and index calculations fit, including byte sizes, strides, affine
  intermediates and loop increments. Reject a program if these conditions or
  the justification for an adopted transformation cannot be established.

The theorem ends at the kernel source under the formal semantics of the
emitted C subset. Acceptance must bind the bytes of that kernel to the program
covered by the proof. Execution on a machine additionally assumes that the C
implementation and hardware realize those semantics. The CLI harness, stdio
and benchmarking are outside the certified artifact.
