//! Tests for rewrite numeric-safety scaffolding.

use vtc_ir::{DType, Dim, Graph, GraphBuilder, NodeId, Shape};
use vtc_rewrite::{Law, NumericSafety, Rewrite, RewriteMode, RuleSet};

struct DummyRewrite {
    name: &'static str,
    laws: &'static [Law],
}

impl Rewrite for DummyRewrite {
    fn name(&self) -> &'static str {
        self.name
    }

    fn laws(&self) -> &'static [Law] {
        self.laws
    }

    fn try_at(&self, graph: &Graph, _node: NodeId) -> Option<Graph> {
        Some(graph.clone())
    }
}

fn sample_graph() -> (Graph, NodeId) {
    let mut builder = GraphBuilder::new();
    let input = builder
        .input("x", DType::F32, Shape::new(vec![Dim::new(1)]))
        .expect("input is structurally valid");
    builder.mark_output(input).expect("output id is valid");
    let graph = builder.build().expect("graph is structurally valid");
    (graph, input)
}

#[test]
fn classifies_laws_by_bit_preservation() {
    let real_only = [
        Law::FloatAddAssoc,
        Law::FloatMulAssoc,
        Law::FloatDistributive,
        Law::AddZeroIdentity,
        Law::MulZeroAnnihilator,
    ];
    for law in real_only {
        assert!(!law.preserves_bits());
    }

    let bit_exact = [
        Law::StructuralOnly,
        Law::IntegerArithmetic,
        Law::FloatAddComm,
        Law::FloatMulComm,
        Law::MulOneIdentity,
        Law::ReluIdempotent,
        Law::NegInvolutive,
    ];
    for law in bit_exact {
        assert!(law.preserves_bits());
    }
}

#[test]
fn derives_safety_from_law_sets() {
    assert_eq!(
        NumericSafety::from_laws(&[Law::StructuralOnly]),
        NumericSafety::BitExact,
    );
    assert_eq!(
        NumericSafety::from_laws(&[Law::FloatAddComm, Law::MulOneIdentity]),
        NumericSafety::BitExact,
    );
    assert_eq!(
        NumericSafety::from_laws(&[Law::FloatAddAssoc]),
        NumericSafety::RealOnly,
    );
    assert_eq!(
        NumericSafety::from_laws(&[Law::FloatAddComm, Law::FloatAddAssoc]),
        NumericSafety::RealOnly,
    );
    assert_eq!(NumericSafety::from_laws(&[]), NumericSafety::RealOnly);
}

#[test]
fn mode_allows_expected_safety_classes() {
    assert!(RewriteMode::Strict.allows(NumericSafety::BitExact));
    assert!(!RewriteMode::Strict.allows(NumericSafety::RealOnly));
    assert!(RewriteMode::FastMath.allows(NumericSafety::BitExact));
    assert!(RewriteMode::FastMath.allows(NumericSafety::RealOnly));
}

#[test]
fn ruleset_filters_rules_by_mode() {
    let mut rules = RuleSet::new();
    rules.add(Box::new(DummyRewrite {
        name: "structural",
        laws: &[Law::StructuralOnly],
    }));
    rules.add(Box::new(DummyRewrite {
        name: "real-only",
        laws: &[Law::FloatAddAssoc],
    }));

    assert_eq!(rules.len(), 2);
    assert_eq!(rules.names(), vec!["structural", "real-only"]);

    let strict_names = rules
        .enabled(RewriteMode::Strict)
        .into_iter()
        .map(Rewrite::name)
        .collect::<Vec<_>>();
    assert_eq!(strict_names, vec!["structural"]);

    let fast_math_names = rules
        .enabled(RewriteMode::FastMath)
        .into_iter()
        .map(Rewrite::name)
        .collect::<Vec<_>>();
    assert_eq!(fast_math_names, vec!["structural", "real-only"]);
}

#[test]
fn boxed_rewrite_is_object_safe() {
    let rule: Box<dyn Rewrite> = Box::new(DummyRewrite {
        name: "structural",
        laws: &[Law::StructuralOnly],
    });
    let (graph, node) = sample_graph();

    assert_eq!(rule.name(), "structural");
    assert_eq!(rule.safety(), NumericSafety::BitExact);
    assert!(rule.try_at(&graph, node).is_some());
}
