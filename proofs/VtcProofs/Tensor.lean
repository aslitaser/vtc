import Mathlib.Algebra.Order.Ring.Rat

namespace VtcProofs

abbrev Tensor (n : Nat) := Fin n -> ℚ

def tneg {n : Nat} (t : Tensor n) : Tensor n :=
  fun i => -(t i)

def trelu {n : Nat} (t : Tensor n) : Tensor n :=
  fun i => max 0 (t i)

def treshape {n m : Nat} (h : n = m) (t : Tensor n) : Tensor m :=
  h ▸ t

end VtcProofs
