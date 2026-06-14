import VtcProofs.Tensor

namespace VtcProofs

theorem neg_neg_rewrite {n : Nat} (t : Tensor n) : tneg (tneg t) = t := by
  funext i
  simp [tneg]

theorem relu_idempotent_rewrite {n : Nat} (t : Tensor n) :
    trelu (trelu t) = trelu t := by
  funext i
  exact max_eq_right (le_max_left (0 : ℚ) (t i))

theorem reshape_fuse_rewrite {n m k : Nat} (h1 : n = m) (h2 : m = k)
    (t : Tensor n) :
    treshape h2 (treshape h1 t) = treshape (h1.trans h2) t := by
  cases h1
  cases h2
  rfl

end VtcProofs
