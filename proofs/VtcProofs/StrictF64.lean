module

public import Init.Data.Float.Model.Float

@[expose] public section

namespace VtcProofs.StrictF64

/-- A binary64 observation. The model stores one representative for all NaNs. -/
abbrev Scalar := Float.Model

/-- A NaN has an all-ones exponent and a nonzero fraction, for either sign. -/
abbrev IsNaNBits (bits : UInt64) : Prop :=
  Float.Model.UnpackedFloat.unpackExponent (spec := .binary64) bits.toBitVec = -1#11 ∧
  Float.Model.UnpackedFloat.unpackMantissa (spec := .binary64) bits.toBitVec ≠ 0#52

/-- Decode raw input bits, preserving the representation of every non-NaN. -/
def decode (bits : UInt64) : Scalar :=
  if h : IsNaNBits bits then Float.Model.nan
  else ⟨bits, ⟨fun exponent fraction => (h ⟨exponent, fraction⟩).elim⟩⟩

theorem toBits_decode_of_not_nan (bits : UInt64) (h : ¬ IsNaNBits bits) :
    (decode bits).toBits = bits := by
  simp [decode, h]

theorem decode_eq_nan_iff (bits : UInt64) :
    decode bits = Float.Model.nan ↔ IsNaNBits bits := by
  by_cases h : IsNaNBits bits
  · exact ⟨fun _ => h, fun _ => dite_eq_left h⟩
  · constructor
    · intro eq
      have bits_eq : bits = Float.Model.nan.toBits := by
        simpa only [toBits_decode_of_not_nan bits h] using congrArg Float.Model.toBits eq
      rw [bits_eq]
      decide
    · exact fun nan => (h nan).elim

/-- Propositional equality of decoded observations keeps signed zeros distinct.
IEEE numerical comparison has different behavior for both zero and NaN. -/
theorem decode_eq_iff (a b : UInt64) :
    decode a = decode b ↔ a = b ∨ (IsNaNBits a ∧ IsNaNBits b) := by
  constructor
  · intro eq
    have sameNaN : IsNaNBits a ↔ IsNaNBits b := by
      rw [← decode_eq_nan_iff a, ← decode_eq_nan_iff b, eq]
    by_cases ha : IsNaNBits a
    · exact Or.inr ⟨ha, sameNaN.mp ha⟩
    · have hb := mt sameNaN.mpr ha
      left
      simpa only [toBits_decode_of_not_nan a ha, toBits_decode_of_not_nan b hb]
        using congrArg Float.Model.toBits eq
  · rintro (rfl | ⟨ha, hb⟩)
    · rfl
    · rw [(decode_eq_nan_iff a).mpr ha, (decode_eq_nan_iff b).mpr hb]

/-- Encoding a model value and decoding it again preserves its observation, even for NaN. -/
@[simp] theorem decode_toBits (x : Scalar) : decode x.toBits = x := by
  obtain ⟨bits, valid⟩ := x
  change decode bits = ⟨bits, valid⟩
  by_cases h : IsNaNBits bits
  · rw [(decode_eq_nan_iff bits).mpr h]
    have eq : bits = Float.Model.nan.toBits :=
      UInt64.toBitVec_inj.mp (valid.eq_packedNaN h.1 h.2)
    subst bits
    rfl
  · simp [decode, h]

/-- Positive zero initializes reductions and is the fallback value for ReLU. -/
def positiveZero : Scalar := decode 0

/-- Each arithmetic operation rounds separately in Lean's logical binary64 model. -/
def add (a b : Scalar) : Scalar := Float.Model.add a b

def sub (a b : Scalar) : Scalar := Float.Model.sub a b

def mul (a b : Scalar) : Scalar := Float.Model.mul a b

def neg (a : Scalar) : Scalar := Float.Model.neg a

/-- Ordered comparison, which is false if either operand is NaN. -/
def lt (a b : Scalar) : Bool := Float.Model.lt a b

def relu (x : Scalar) : Scalar := if lt positiveZero x then x else positiveZero

-- Distinct zeros, and a signaling NaN and a quiet NaN with different signs and payloads.
example : decode 0 ≠ decode 0x8000000000000000 := by decide
example : decode 0x7ff0000000000001 = decode 0xfff8000000000042 := by decide

-- Half an ulp at 1.0 rounds to the neighbor with an even significand in both directions.
example : (add (decode 0x3ff0000000000000) (decode 0x3ca0000000000000)).toBits =
    0x3ff0000000000000 := by decide
example : (add (decode 0x3ff0000000000001) (decode 0x3ca0000000000000)).toBits =
    0x3ff0000000000002 := by decide

-- Cancellation at the normal/subnormal boundary, and overflow to positive infinity.
example : (sub (decode 0x0010000000000000) (decode 0x000fffffffffffff)).toBits = 1 := by decide
example : (mul (decode 0x7fefffffffffffff) (decode 0x4000000000000000)).toBits =
    0x7ff0000000000000 := by decide

example : (neg positiveZero).toBits = 0x8000000000000000 := by decide
example : (relu (decode 0x8000000000000000)).toBits = 0 := by decide
example : (relu (decode 0x7ff8000000000001)).toBits = 0 := by decide
example : (relu (decode 1)).toBits = 1 := by decide

end VtcProofs.StrictF64
