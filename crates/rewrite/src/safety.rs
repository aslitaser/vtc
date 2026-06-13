//! Numeric-safety classification for graph rewrites.

/// Algebraic law used to justify a rewrite.
///
/// Each law records whether it is target bit-exact for IEEE-754 floats and
/// target integers. The list is intentionally extensible as the rewrite system
/// grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// Pure dataflow or graph-structure change with no value change.
    ///
    /// Target bit-exact: yes.
    StructuralOnly,

    /// Exact integer associativity, commutativity, or distributivity.
    ///
    /// Target bit-exact: yes. Caveat: the reference model treats integers as
    /// unbounded; hardware overflow or wrapping is outside the current model.
    IntegerArithmetic,

    /// Floating addition commutativity, `a + b == b + a`.
    ///
    /// Target bit-exact: yes, ignoring NaN-payload choice.
    FloatAddComm,

    /// Floating multiplication commutativity, `a * b == b * a`.
    ///
    /// Target bit-exact: yes, ignoring NaN-payload choice.
    FloatMulComm,

    /// Multiplicative identity, `x * 1 == x`.
    ///
    /// Target bit-exact: yes, including signed zero and infinities; NaN-payload
    /// choice is ignored.
    MulOneIdentity,

    /// Floating addition associativity, `(a + b) + c == a + (b + c)`.
    ///
    /// Target bit-exact: no. This is the canonical non-bit-exact floating law.
    FloatAddAssoc,

    /// Floating multiplication associativity, `(a * b) * c == a * (b * c)`.
    ///
    /// Target bit-exact: no.
    FloatMulAssoc,

    /// Floating distributivity, `a * (b + c) == a * b + a * c`.
    ///
    /// Target bit-exact: no.
    FloatDistributive,

    /// Additive identity, `x + 0 == x`.
    ///
    /// Target bit-exact: no. Caveat: `(-0.0) + (+0.0) = +0.0`, so signed zero
    /// breaks bit identity.
    AddZeroIdentity,

    /// Multiplicative zero annihilator, `x * 0 == 0`.
    ///
    /// Target bit-exact: no. Caveat: `inf * 0` and `NaN * 0` are NaN, and the
    /// sign of zero depends on `x`.
    MulZeroAnnihilator,

    /// `ReLU` idempotence, `relu(relu(x)) == relu(x)`.
    ///
    /// Target bit-exact: yes under the current `ReLU` semantics, where
    /// `relu(x) = x` if `x > 0` and `0` otherwise.
    ReluIdempotent,

    /// Negation involution, `-(-x) == x`.
    ///
    /// Target bit-exact: yes. Negation toggles the sign bit; doing it twice
    /// restores the exact bit pattern, including signed zero and NaN.
    NegInvolutive,
}

impl Law {
    /// Returns true iff a rewrite justified solely by this law yields
    /// bit-identical results on the target.
    ///
    /// The target model is IEEE-754 floats plus target integers. This law list
    /// is extensible as new rewrite justifications are introduced.
    #[must_use]
    pub const fn preserves_bits(&self) -> bool {
        match self {
            Self::StructuralOnly
            | Self::IntegerArithmetic
            | Self::FloatAddComm
            | Self::FloatMulComm
            | Self::MulOneIdentity
            | Self::ReluIdempotent
            | Self::NegInvolutive => true,
            Self::FloatAddAssoc
            | Self::FloatMulAssoc
            | Self::FloatDistributive
            | Self::AddZeroIdentity
            | Self::MulZeroAnnihilator => false,
        }
    }
}

/// Derived numeric-safety class for a rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericSafety {
    /// The rewrite is target bit-exact.
    BitExact,
    /// The rewrite is only justified by the exact-rational reference model.
    RealOnly,
}

impl NumericSafety {
    /// Derives numeric safety from the laws that justify a rewrite.
    ///
    /// An empty law list is conservatively classified as [`Self::RealOnly`];
    /// this avoids treating an unjustified rule as bit-exact through vacuous
    /// `all` semantics.
    #[must_use]
    pub fn from_laws(laws: &[Law]) -> Self {
        if laws.is_empty() {
            Self::RealOnly
        } else if laws.iter().all(Law::preserves_bits) {
            Self::BitExact
        } else {
            Self::RealOnly
        }
    }
}

/// Rewrite enablement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteMode {
    /// IEEE-bit-faithful mode; only bit-exact rewrites are allowed.
    Strict,
    /// Future fast-math mode; both bit-exact and real-only rewrites are allowed.
    FastMath,
}

impl RewriteMode {
    /// Returns whether this mode allows a rewrite with the given safety class.
    ///
    /// `FastMath` corresponds to a future `--fast-math` CLI flag. Under
    /// [`Self::Strict`], the compiler must produce IEEE-bit-faithful results.
    #[must_use]
    pub const fn allows(&self, safety: NumericSafety) -> bool {
        match self {
            Self::Strict => matches!(safety, NumericSafety::BitExact),
            Self::FastMath => true,
        }
    }
}
