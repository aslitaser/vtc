module

public import VtcProofs.StrictF64
public import Init.Data.List.Lemmas
public import Init.Data.List.Nat.Basic
public import Init.Omega

@[expose] public section

namespace VtcProofs.StrictGraph

abbrev Shape := List Nat

def numel (shape : Shape) : Nat := shape.prod

/-- Rank zero is one scalar; all explicit extents are positive and the element
count fits an unsigned 64-bit word. -/
def ValidShape (shape : Shape) : Prop :=
  (∀ dim ∈ shape, 0 < dim) ∧ numel shape < 2 ^ 64

instance (shape : Shape) : Decidable (ValidShape shape) := inferInstanceAs
  (Decidable ((∀ dim ∈ shape, 0 < dim) ∧ numel shape < 2 ^ 64))

def FiniteBits (bits : UInt64) : Prop :=
  Float.Model.UnpackedFloat.unpackExponent (spec := .binary64) bits.toBitVec ≠ -1#11

instance (bits : UInt64) : Decidable (FiniteBits bits) := inferInstanceAs
  (Decidable (Float.Model.UnpackedFloat.unpackExponent (spec := .binary64) bits.toBitVec ≠ -1#11))

inductive RawNode where
  | input (name : String) (shape : List Nat)
  | const (shape : List Nat) (bits : List UInt64)
  | add (lhs rhs : Nat)
  | relu (src : Nat)
  deriving Repr, DecidableEq

structure RawGraph where
  nodes : List RawNode
  outputs : List Nat
  deriving Repr, DecidableEq

def nextNames (names : List String) : RawNode → List String
  | .input name _ => name :: names
  | _ => names

/-- A node is indexed by its original syntax and its inferred shape. Every
reference is bounded by the already checked prefix, so forward edges are absent. -/
inductive TypedNode (shapes : List Shape) (names : List String) : RawNode → Shape → Type where
  | input (name : String) (shape : Shape) (valid : ValidShape shape) (fresh : name ∉ names) :
      TypedNode shapes names (.input name shape) shape
  | const (shape : Shape) (bits : List UInt64) (valid : ValidShape shape)
      (length_eq : bits.length = numel shape) (finite : ∀ bit ∈ bits, FiniteBits bit) :
      TypedNode shapes names (.const shape bits) shape
  | add (lhs rhs : Fin shapes.length) (same : shapes.get lhs = shapes.get rhs) :
      TypedNode shapes names (.add lhs.val rhs.val) (shapes.get lhs)
  | relu (src : Fin shapes.length) :
      TypedNode shapes names (.relu src.val) (shapes.get src)

/-- The certificate follows the original node list and records each node once.
It does not expand a DAG into an expression tree. -/
inductive Program : List Shape → List String → List RawNode → List Shape → Type where
  | done {shapes names} : Program shapes names [] shapes
  | step {shapes names node shape rest finalShapes}
      (head : TypedNode shapes names node shape)
      (tail : Program (shapes ++ [shape]) (nextNames names node) rest finalShapes) :
      Program shapes names (node :: rest) finalShapes

theorem Program.length_eq {shapes names nodes finalShapes}
    (program : Program shapes names nodes finalShapes) :
    finalShapes.length = shapes.length + nodes.length := by
  induction program with
  | done => simp
  | step head tail ih =>
      simp only [List.length_append, List.length_cons, List.length_nil] at ih ⊢
      omega

def OutputsValid (shapes : List Shape) (outputs : List Nat) : Prop :=
  outputs ≠ [] ∧ ∀ id ∈ outputs, id < shapes.length

instance (shapes : List Shape) (outputs : List Nat) : Decidable (OutputsValid shapes outputs) :=
  inferInstanceAs (Decidable (outputs ≠ [] ∧ ∀ id ∈ outputs, id < shapes.length))

/-- Independent intrinsic typing evidence for the same raw graph. -/
def WellTyped (raw : RawGraph) : Prop :=
  ∃ shapes, Nonempty (Program [] [] raw.nodes shapes) ∧ OutputsValid shapes raw.outputs

structure CheckedGraph where
  raw : RawGraph
  shapes : List Shape
  program : Program [] [] raw.nodes shapes
  outputsValid : OutputsValid shapes raw.outputs

theorem CheckedGraph.wellTyped (graph : CheckedGraph) : WellTyped graph.raw :=
  ⟨graph.shapes, ⟨graph.program⟩, graph.outputsValid⟩

theorem CheckedGraph.shapes_length (graph : CheckedGraph) :
    graph.shapes.length = graph.raw.nodes.length := by
  simpa using graph.program.length_eq

theorem CheckedGraph.output_in_bounds (graph : CheckedGraph) {id : Nat}
    (selected : id ∈ graph.raw.outputs) : id < graph.raw.nodes.length := by
  rw [← graph.shapes_length]
  exact graph.outputsValid.2 id selected

structure CheckedNode (shapes : List Shape) (names : List String) (raw : RawNode) where
  shape : Shape
  typed : TypedNode shapes names raw shape

def checkNode (shapes : List Shape) (names : List String) (raw : RawNode) :
    Except String (CheckedNode shapes names raw) :=
  match raw with
  | .input name shape =>
      if valid : ValidShape shape then
        if fresh : name ∉ names then .ok ⟨shape, .input name shape valid fresh⟩
        else .error "duplicate input name"
      else .error "invalid input shape"
  | .const shape bits =>
      if valid : ValidShape shape then
        if length_eq : bits.length = numel shape then
          -- Traverse the payload explicitly: UInt64 also has a finite-domain
          -- universal decision procedure, which would enumerate all 2^64 words.
          if finite : bits.all (fun bit => decide (FiniteBits bit)) = true then
            .ok ⟨shape, .const shape bits valid length_eq (by
              intro bit member
              exact of_decide_eq_true ((List.all_eq_true.mp finite) bit member))⟩
          else .error "non-finite constant"
        else .error "constant payload length differs from shape"
      else .error "invalid constant shape"
  | .add lhs rhs =>
      if left : lhs < shapes.length then
        if right : rhs < shapes.length then
          if same : shapes.get ⟨lhs, left⟩ = shapes.get ⟨rhs, right⟩ then
            .ok ⟨_, .add ⟨lhs, left⟩ ⟨rhs, right⟩ same⟩
          else .error "add shapes differ"
        else .error "add rhs is not a backward reference"
      else .error "add lhs is not a backward reference"
  | .relu src =>
      if bound : src < shapes.length then .ok ⟨_, .relu ⟨src, bound⟩⟩
      else .error "relu source is not a backward reference"

structure CheckedProgram (shapes : List Shape) (names : List String) (raw : List RawNode) where
  finalShapes : List Shape
  program : Program shapes names raw finalShapes

def checkNodes (shapes : List Shape) (names : List String) (raw : List RawNode) :
    Except String (CheckedProgram shapes names raw) :=
  match raw with
  | [] => .ok ⟨shapes, .done⟩
  | node :: rest => do
      let head ← checkNode shapes names node
      let tail ← checkNodes (shapes ++ [head.shape]) (nextNames names node) rest
      pure ⟨tail.finalShapes, .step head.typed tail.program⟩

def check (raw : RawGraph) : Except String CheckedGraph := do
  let checked ← checkNodes [] [] raw.nodes
  if valid : OutputsValid checked.finalShapes raw.outputs then
    pure ⟨raw, checked.finalShapes, checked.program, valid⟩
  else .error "outputs must be nonempty backward references"

theorem check_sound {raw : RawGraph} {checked : CheckedGraph}
    (accepted : check raw = .ok checked) : checked.raw = raw ∧ WellTyped raw := by
  cases nodes : checkNodes [] [] raw.nodes with
  | error message => simp [check, nodes, bind, Except.bind] at accepted
  | ok program =>
      by_cases valid : OutputsValid program.finalShapes raw.outputs
      · simp [check, nodes, valid, bind, Except.bind, pure, Except.pure] at accepted
        subst checked
        exact ⟨rfl, program.finalShapes, ⟨program.program⟩, valid⟩
      · simp [check, nodes, valid, bind, Except.bind] at accepted

structure Tensor (shape : Shape) where
  data : List StrictF64.Scalar
  length_eq : data.length = numel shape

def tensorAdd {leftShape rightShape : Shape} (same : leftShape = rightShape)
    (left : Tensor leftShape) (right : Tensor rightShape) : Tensor leftShape where
  data := List.zipWith StrictF64.add left.data right.data
  length_eq := by
    simp only [List.length_zipWith, left.length_eq, right.length_eq, same, Nat.min_self]

def tensorRelu {shape : Shape} (input : Tensor shape) : Tensor shape where
  data := input.data.map StrictF64.relu
  length_eq := by simp [input.length_eq]

abbrev Inputs := String → List StrictF64.Scalar

def NodeInputMatches (inputs : Inputs) : RawNode → Prop
  | .input name shape => (inputs name).length = numel shape
  | _ => True

instance (inputs : Inputs) (node : RawNode) : Decidable (NodeInputMatches inputs node) := by
  cases node <;> unfold NodeInputMatches <;> infer_instance

def MatchesNodes (nodes : List RawNode) (inputs : Inputs) : Prop :=
  ∀ node ∈ nodes, NodeInputMatches inputs node

instance (nodes : List RawNode) (inputs : Inputs) : Decidable (MatchesNodes nodes inputs) :=
  inferInstanceAs (Decidable (∀ node ∈ nodes, NodeInputMatches inputs node))

def InputsMatch (raw : RawGraph) (inputs : Inputs) : Prop := MatchesNodes raw.nodes inputs

instance (raw : RawGraph) (inputs : Inputs) : Decidable (InputsMatch raw inputs) :=
  inferInstanceAs (Decidable (MatchesNodes raw.nodes inputs))

inductive Values : List Shape → Type where
  | nil : Values []
  | cons {shape shapes} (head : Tensor shape) (tail : Values shapes) : Values (shape :: shapes)

def Values.get : {shapes : List Shape} → Values shapes → (id : Fin shapes.length) →
    Tensor (shapes.get id)
  | [], .nil, id => Fin.elim0 id
  | _ :: _, .cons head _, ⟨0, _⟩ => head
  | _ :: _, .cons _ tail, ⟨id + 1, bound⟩ => tail.get ⟨id, by simpa using bound⟩

def Values.snoc {shape : Shape} :
    {shapes : List Shape} → Values shapes → Tensor shape → Values (shapes ++ [shape])
  | [], .nil, value => .cons value .nil
  | _ :: _, .cons head tail, value => .cons head (tail.snoc value)

def Values.buffers : {shapes : List Shape} → Values shapes → List (List StrictF64.Scalar)
  | [], .nil => []
  | _ :: _, .cons head tail => head.data :: tail.buffers

@[simp] theorem Values.buffers_snoc {shapes : List Shape} {shape : Shape} (values : Values shapes)
    (value : Tensor shape) : (values.snoc value).buffers = values.buffers ++ [value.data] := by
  induction values with
  | nil => rfl
  | cons head tail ih => simp [Values.snoc, Values.buffers, ih]

theorem Values.get_buffer {shapes : List Shape} (values : Values shapes)
    (id : Fin shapes.length) : values.buffers[id.val]? = some (values.get id).data := by
  induction values with
  | nil => exact Fin.elim0 id
  | cons head tail ih =>
      rcases id with ⟨id, bound⟩
      cases id with
      | zero => rfl
      | succ id => exact ih ⟨id, by simpa using bound⟩

def TypedNode.eval {shapes names node shape} (typed : TypedNode shapes names node shape)
    (values : Values shapes) (inputs : Inputs)
    (matching : ∀ name inputShape, node = .input name inputShape →
      (inputs name).length = numel inputShape) : Tensor shape :=
  match typed with
  | .input name shape _ _ => ⟨inputs name, matching name shape rfl⟩
  | .const _ bits _ length_eq _ =>
      ⟨bits.map StrictF64.decode, by simpa using length_eq⟩
  | .add lhs rhs same => tensorAdd same (values.get lhs) (values.get rhs)
  | .relu src => tensorRelu (values.get src)

/-- Raw operational semantics over stored scalar buffers, independent of the
typing certificate. Arithmetic uses the strict logical binary64 operations. -/
inductive NodeEval (inputs : Inputs) (stored : List (List StrictF64.Scalar)) :
    RawNode → List StrictF64.Scalar → Prop where
  | input (name shape) : NodeEval inputs stored (.input name shape) (inputs name)
  | const (shape bits) :
      NodeEval inputs stored (.const shape bits) (bits.map StrictF64.decode)
  | add (lhs rhs : Nat) (left right : List StrictF64.Scalar)
      (left_at : stored[lhs]? = some left) (right_at : stored[rhs]? = some right) :
      NodeEval inputs stored (.add lhs rhs) (List.zipWith StrictF64.add left right)
  | relu (src : Nat) (input : List StrictF64.Scalar) (source_at : stored[src]? = some input) :
      NodeEval inputs stored (.relu src) (input.map StrictF64.relu)

inductive EvalNodes (inputs : Inputs) :
    List (List StrictF64.Scalar) → List RawNode → List (List StrictF64.Scalar) → Prop where
  | done (stored) : EvalNodes inputs stored [] stored
  | step {stored node value rest result}
      (head : NodeEval inputs stored node value)
      (tail : EvalNodes inputs (stored ++ [value]) rest result) :
      EvalNodes inputs stored (node :: rest) result

theorem TypedNode.eval_semantics {shapes names node shape}
    (typed : TypedNode shapes names node shape) (values : Values shapes) (inputs : Inputs)
    (matching : ∀ name inputShape, node = .input name inputShape →
      (inputs name).length = numel inputShape) :
    NodeEval inputs values.buffers node (typed.eval values inputs matching).data := by
  cases typed with
  | input name shape valid fresh => exact .input name shape
  | const shape bits valid length_eq finite => exact .const shape bits
  | add lhs rhs same => exact .add _ _ _ _ (values.get_buffer lhs) (values.get_buffer rhs)
  | relu src => exact .relu _ _ (values.get_buffer src)

def Program.eval {shapes names nodes finalShapes}
    (program : Program shapes names nodes finalShapes) (values : Values shapes) (inputs : Inputs)
    (matching : MatchesNodes nodes inputs) : Values finalShapes :=
  match program with
  | .done => values
  | .step head tail =>
      let value := head.eval values inputs (by
        intro name shape eq
        exact matching (.input name shape) (by simp [← eq]))
      tail.eval (values.snoc value) inputs (by
        intro node mem
        exact matching node (List.mem_cons_of_mem _ mem))

theorem Program.eval_semantics {shapes names nodes finalShapes}
    (program : Program shapes names nodes finalShapes) (values : Values shapes) (inputs : Inputs)
    (matching : MatchesNodes nodes inputs) :
    EvalNodes inputs values.buffers nodes (program.eval values inputs matching).buffers := by
  induction program with
  | done => exact .done _
  | step head tail ih =>
      simp only [Program.eval]
      apply EvalNodes.step (head.eval_semantics values inputs _)
      rw [← Values.buffers_snoc]
      exact ih _ _

def evalValues (graph : CheckedGraph) (inputs : Inputs) (matching : InputsMatch graph.raw inputs) :
    Values graph.shapes := graph.program.eval .nil inputs matching

def selectOutputs {shapes : List Shape} (values : Values shapes) (outputs : List Nat)
    (valid : ∀ id ∈ outputs, id < shapes.length) : List (List StrictF64.Scalar) :=
  match outputs with
  | [] => []
  | id :: rest =>
      (values.get ⟨id, valid id (by simp)⟩).data ::
        selectOutputs values rest (by
          intro next mem
          exact valid next (List.mem_cons_of_mem _ mem))

/-- Total evaluation for every matching input. Selection follows the original
output list, so order and duplicate references are retained. -/
def eval (graph : CheckedGraph) (inputs : Inputs) (matching : InputsMatch graph.raw inputs) :
    List (List StrictF64.Scalar) :=
  selectOutputs (evalValues graph inputs matching) graph.raw.outputs graph.outputsValid.2

inductive OutputSelection (buffers : List (List StrictF64.Scalar)) :
    List Nat → List (List StrictF64.Scalar) → Prop where
  | nil : OutputSelection buffers [] []
  | cons {id value ids values} (head : buffers[id]? = some value)
      (tail : OutputSelection buffers ids values) :
      OutputSelection buffers (id :: ids) (value :: values)

theorem selectOutputs_selection {shapes : List Shape} (values : Values shapes)
    (outputs : List Nat) (valid : ∀ id ∈ outputs, id < shapes.length) :
    OutputSelection values.buffers outputs (selectOutputs values outputs valid) := by
  induction outputs with
  | nil => exact .nil
  | cons id rest ih => exact .cons (values.get_buffer ⟨id, valid id (by simp)⟩) (ih _)

theorem eval_semantics (graph : CheckedGraph) (inputs : Inputs)
    (matching : InputsMatch graph.raw inputs) :
    EvalNodes inputs [] graph.raw.nodes (evalValues graph inputs matching).buffers ∧
      OutputSelection (evalValues graph inputs matching).buffers
        graph.raw.outputs (eval graph inputs matching) :=
  ⟨graph.program.eval_semantics .nil inputs matching,
    selectOutputs_selection _ _ _⟩

theorem selection_at {buffers : List (List StrictF64.Scalar)}
    {ids : List Nat} {outputs : List (List StrictF64.Scalar)}
    (selection : OutputSelection buffers ids outputs)
    {index id : Nat} (position : ids[index]? = some id) :
    ∃ value, outputs[index]? = some value ∧ buffers[id]? = some value := by
  induction selection generalizing index with
  | nil => simp at position
  | @cons next value ids outputs head tail ih =>
      cases index with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at position
          subst id
          exact ⟨value, rfl, head⟩
      | succ index => exact ih position

/-- Every original raw output position resolves to its selected evaluated node,
including repeated IDs. This connects outputs to values, not only to shapes. -/
theorem eval_output (graph : CheckedGraph) (inputs : Inputs)
    (matching : InputsMatch graph.raw inputs) {index id : Nat}
    (position : graph.raw.outputs[index]? = some id) :
    ∃ value, (eval graph inputs matching)[index]? = some value ∧
      (evalValues graph inputs matching).buffers[id]? = some value :=
  selection_at (eval_semantics graph inputs matching).2 position

def selectShapes (shapes : List Shape) (outputs : List Nat)
    (valid : ∀ id ∈ outputs, id < shapes.length) : List Shape :=
  match outputs with
  | [] => []
  | id :: rest =>
      shapes.get ⟨id, valid id (by simp)⟩ ::
        selectShapes shapes rest (by
          intro next mem
          exact valid next (List.mem_cons_of_mem _ mem))

/-- Logical output dimensions in original reference order, including duplicates. -/
def outputShapes (graph : CheckedGraph) : List Shape :=
  selectShapes graph.shapes graph.raw.outputs graph.outputsValid.2

theorem selectShapes_at (shapes : List Shape) (outputs : List Nat)
    (valid : ∀ id ∈ outputs, id < shapes.length) {index id : Nat}
    (position : outputs[index]? = some id) :
    ∃ shape, (selectShapes shapes outputs valid)[index]? = some shape ∧
      shapes[id]? = some shape := by
  induction outputs generalizing index with
  | nil => simp at position
  | cons next rest ih =>
      cases index with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at position
          subst id
          refine ⟨shapes.get ⟨next, valid next (by simp)⟩, rfl, ?_⟩
          exact List.getElem?_eq_some_iff.mpr ⟨_, rfl⟩
      | succ index => exact ih _ position

theorem outputShapes_at (graph : CheckedGraph) {index id : Nat}
    (position : graph.raw.outputs[index]? = some id) :
    ∃ shape, (outputShapes graph)[index]? = some shape ∧ graph.shapes[id]? = some shape :=
  selectShapes_at _ _ _ position

/-- Both the original node's logical shape and its evaluated scalar buffer are
selected at each output position. The buffer has exactly that shape's size. -/
theorem eval_output_typed (graph : CheckedGraph) (inputs : Inputs)
    (matching : InputsMatch graph.raw inputs) {index id : Nat}
    (position : graph.raw.outputs[index]? = some id) :
    ∃ shape value,
      (outputShapes graph)[index]? = some shape ∧
      graph.shapes[id]? = some shape ∧
      (eval graph inputs matching)[index]? = some value ∧
      (evalValues graph inputs matching).buffers[id]? = some value ∧
      value.length = numel shape := by
  obtain ⟨shape, shape_at, shape_stored⟩ := outputShapes_at graph position
  obtain ⟨value, value_at, value_stored⟩ := eval_output graph inputs matching position
  let ref : Fin graph.shapes.length :=
    ⟨id, graph.outputsValid.2 id (List.mem_of_getElem? position)⟩
  have same_shape : graph.shapes.get ref = shape :=
    (List.getElem?_eq_some_iff.mp shape_stored).2
  have same_value : ((evalValues graph inputs matching).get ref).data = value := by
    exact Option.some.inj ((Values.get_buffer _ ref).symm.trans value_stored)
  refine ⟨shape, value, shape_at, shape_stored, value_at, value_stored, ?_⟩
  rw [← same_value, ← same_shape]
  exact Tensor.length_eq _

theorem OutputSelection.length_eq {buffers : List (List StrictF64.Scalar)}
    {ids : List Nat} {outputs : List (List StrictF64.Scalar)}
    (selection : OutputSelection buffers ids outputs) : outputs.length = ids.length := by
  induction selection with
  | nil => rfl
  | cons head tail ih => simp [ih]

theorem eval_length (graph : CheckedGraph) (inputs : Inputs)
    (matching : InputsMatch graph.raw inputs) :
    (eval graph inputs matching).length = graph.raw.outputs.length :=
  (eval_semantics graph inputs matching).2.length_eq

end VtcProofs.StrictGraph
