//! Rewrite-rule container and mode filtering.

use crate::{Rewrite, RewriteMode};

/// Container for rewrite rules.
#[derive(Default)]
pub struct RuleSet {
    rules: Vec<Box<dyn Rewrite>>,
}

impl RuleSet {
    /// Creates an empty rule set.
    #[must_use]
    pub const fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a rule to the set.
    pub fn add(&mut self, rule: Box<dyn Rewrite>) {
        self.rules.push(rule);
    }

    /// Returns the number of rules in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Returns whether the set contains no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Returns rule names in insertion order.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.rules.iter().map(|rule| rule.name()).collect()
    }

    /// Returns rules enabled by `mode`, preserving insertion order.
    #[must_use]
    pub fn enabled(&self, mode: RewriteMode) -> Vec<&dyn Rewrite> {
        self.rules
            .iter()
            .map(Box::as_ref)
            .filter(|rule| mode.allows(rule.safety()))
            .collect()
    }
}
