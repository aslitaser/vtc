//! Schedule legality modes.

/// Legality mode for schedule transformations.
///
/// This mirrors `vtc-rewrite`'s rewrite mode for now. A later cleanup can move
/// the shared notion into a common crate. [`Mode::Strict`] preserves IEEE `f64`
/// bits; [`Mode::FastMath`] preserves only the exact-rational result and may
/// reorder associative reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Preserve exact-rational results and IEEE `f64` bits.
    Strict,
    /// Preserve exact-rational results while allowing floating-point bit drift.
    FastMath,
}
