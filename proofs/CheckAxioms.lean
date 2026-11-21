import Lean
import Lake.Config.Glob
import VtcProofs

open Lean

-- Run from proofs/ after building VtcProofs. Audit by defining module so that
-- declarations in other namespaces (including root-level axioms) are covered.
run_cmd do
  let env ← getEnv
  let modules := env.allImportedModuleNames
  let library := Lake.Glob.andSubmodules `VtcProofs
  library.forEachModuleIn "." fun mod => do
    unless modules.contains mod do
      throwError "proof module {mod} is missing from the imports of VtcProofs.lean"

  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut count := 0
  for (decl, _) in env.constants do
    let some idx := env.getModuleIdxFor? decl | continue
    unless library.matches modules[idx.toNat]! do continue
    count := count + 1
    for ax in ← collectAxioms decl do
      unless allowed.contains ax do
        logError m!"{decl} depends on unapproved axiom {ax}"
  if count == 0 then
    throwError "no VtcProofs declarations found to audit"
