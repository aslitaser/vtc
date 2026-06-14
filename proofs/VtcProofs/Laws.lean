import Mathlib.Algebra.Order.Ring.Rat

namespace VtcProofs

theorem law_float_add_comm (a b : ℚ) : a + b = b + a :=
  add_comm a b

-- This law is true over Q, but it is real-only for IEEE floats because
-- reassociation can change rounding.
theorem law_float_add_assoc (a b c : ℚ) : (a + b) + c = a + (b + c) :=
  add_assoc a b c

end VtcProofs
