//! Rewrite-rule trait.

use vtc_ir::{Graph, NodeId};

use crate::{Law, NumericSafety};

/// A graph rewrite rule.
///
/// Real rules must declare a non-empty law list. Structural rules declare
/// `&[Law::StructuralOnly]`. A rule cannot independently assert its numeric
/// safety; [`Self::safety`] is derived from [`Self::laws`].
pub trait Rewrite {
    /// Returns a stable rule name.
    fn name(&self) -> &'static str;

    /// Returns the algebraic laws that justify this rule.
    fn laws(&self) -> &'static [Law];

    /// Returns this rule's derived numeric-safety class.
    fn safety(&self) -> NumericSafety {
        NumericSafety::from_laws(self.laws())
    }

    /// Attempts to rewrite `graph` at `node`.
    ///
    /// Returns the full rewritten graph if the rule fires, or `None` if it does
    /// not apply. Graph-surgery helpers are intentionally left for a later step.
    fn try_at(&self, graph: &Graph, node: NodeId) -> Option<Graph>;
}
