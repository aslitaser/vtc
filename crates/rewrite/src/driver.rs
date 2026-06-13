//! Deterministic rewrite fixpoint driver.

#![allow(
    clippy::module_name_repetitions,
    reason = "step-9 public API names intentionally include DriverConfig"
)]

use vtc_ir::Graph;

use crate::{Rewrite, RewriteMode, RuleSet, SurgeryError, prune_and_rebuild};

/// Configuration for a rewrite-driver run.
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// Rewrite mode used to select enabled rules.
    pub mode: RewriteMode,
    /// Maximum number of rewrites to apply before stopping.
    pub max_steps: usize,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            mode: RewriteMode::Strict,
            max_steps: 1_000,
        }
    }
}

/// One rewrite applied by the driver.
#[derive(Debug, Clone)]
pub struct AppliedRewrite {
    /// Stable rule name that fired.
    pub rule: &'static str,
}

/// Result of a rewrite-driver run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Final canonical graph.
    pub graph: Graph,
    /// Rewrite sequence applied in order.
    pub applied: Vec<AppliedRewrite>,
    /// Whether a complete no-firing scan confirmed a true fixpoint.
    pub reached_fixpoint: bool,
}

impl RunResult {
    /// Returns the number of rewrite steps applied.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.applied.len()
    }
}

/// Runs enabled rewrites to a fixpoint or until the step cap is reached.
///
/// The input graph is structurally validated and canonicalized with
/// [`prune_and_rebuild`] before rewriting. Each step scans nodes by ascending
/// node id and rules in [`RuleSet::enabled`] order, applies exactly one rewrite,
/// then canonicalizes again. Termination for arbitrary rule sets is guaranteed
/// by [`DriverConfig::max_steps`]. A `true` [`RunResult::reached_fixpoint`]
/// means a full deterministic scan found no firing rule; a `false` value means
/// the cap stopped the run before a fixpoint was confirmed.
///
/// # Errors
///
/// Returns [`SurgeryError`] if the input or any intermediate graph is
/// structurally invalid.
pub fn run(graph: &Graph, rules: &RuleSet, cfg: &DriverConfig) -> Result<RunResult, SurgeryError> {
    graph.validate_structure()?;
    let mut graph = prune_and_rebuild(graph)?;
    let enabled = rules.enabled(cfg.mode);
    let mut applied = Vec::new();
    let reached_fixpoint;

    loop {
        let Some((rule, rewritten)) = find_first_firing(&graph, &enabled) else {
            reached_fixpoint = true;
            break;
        };
        if applied.len() >= cfg.max_steps {
            reached_fixpoint = false;
            break;
        }

        graph = prune_and_rebuild(&rewritten)?;
        applied.push(AppliedRewrite { rule });
    }

    Ok(RunResult {
        graph,
        applied,
        reached_fixpoint,
    })
}

fn find_first_firing(graph: &Graph, rules: &[&dyn Rewrite]) -> Option<(&'static str, Graph)> {
    let mut node_ids = graph.topo_order().ok()?;
    node_ids.sort_by_key(|id| id.index());
    for node in node_ids {
        for &rule in rules {
            if let Some(rewritten) = rule.try_at(graph, node) {
                return Some((rule.name(), rewritten));
            }
        }
    }
    None
}
