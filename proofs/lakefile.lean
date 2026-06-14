import Lake
open Lake DSL

package «vtc-proofs» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0-rc2"

@[default_target]
lean_lib VtcProofs where
