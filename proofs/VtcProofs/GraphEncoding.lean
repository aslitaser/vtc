module

public import VtcProofs.Graph
public import Lean.Data.Json.Parser

@[expose] public section

namespace VtcProofs.StrictGraph

open Lean

def decodeNat (json : Json) : Except String Nat :=
  json.getNat?

def decodeShape (json : Json) : Except String (List Nat) := do
  let values ← json.getArr?
  return (← values.mapM decodeNat).toList

def hexDigit (char : Char) : Except String Nat :=
  if '0' ≤ char ∧ char ≤ '9' then
    .ok (char.toNat - '0'.toNat)
  else if 'a' ≤ char ∧ char ≤ 'f' then
    .ok (char.toNat - 'a'.toNat + 10)
  else
    .error "binary64 bits must use lowercase hexadecimal digits"

/-- The wire format uses exactly sixteen lowercase hexadecimal digits, without
a prefix. Decode via unbounded naturals before constructing the machine word. -/
def decodeBits (json : Json) : Except String UInt64 := do
  let text ← json.getStr?
  unless text.length == 16 do
    throw "binary64 bits must contain exactly sixteen lowercase hexadecimal digits"
  let mut value : Nat := 0
  for char in text.toList do
    value := 16 * value + (← hexDigit char)
  if value < 2 ^ 64 then
    return UInt64.ofNat value
  else
    throw "binary64 bits exceed UInt64"

def decodeNode (json : Json) : Except String RawNode := do
  match json with
  | .arr #[.str "input", .str "f64", .str name, shape] =>
    return .input name (← decodeShape shape)
  | .arr #[.str "const", .str "f64", shape, .arr values] =>
    return .const (← decodeShape shape) (← values.mapM decodeBits).toList
  | .arr #[.str "add", lhs, rhs] =>
    return .add (← decodeNat lhs) (← decodeNat rhs)
  | .arr #[.str "relu", source] =>
    return .relu (← decodeNat source)
  | _ => throw "invalid node: unsupported operation, dtype, field type or arity"

/-- Decode the versioned JSON structure without trusting it to be a valid graph.
`Json.parse` consumes the entire text and rejects trailing non-whitespace. -/
def parseRaw (text : String) : Except String RawGraph := do
  let json ← Json.parse text
  match json with
  | .arr #[.str "vtc.graph", version, .arr nodes, .arr outputs] =>
    unless (← decodeNat version) == 1 do
      throw "unsupported vtc.graph version"
    return {
      nodes := (← nodes.mapM decodeNode).toList
      outputs := (← outputs.mapM decodeNat).toList
    }
  | _ => throw "expected [\"vtc.graph\",1,[nodes],[outputs]]"

/-- The entry point accepts only parsed graphs accepted by the independent
well-formedness checker. Neither JSON decoding nor the producer asserts validity. -/
def decode (text : String) : Except String CheckedGraph := do
  check (← parseRaw text)

/-- The file boundary rejects invalid UTF-8 rather than replacing invalid bytes. -/
def decodeBytes (bytes : ByteArray) : Except String CheckedGraph :=
  match String.fromUTF8? bytes with
  | none => .error "graph file is not valid UTF-8"
  | some text => decode text

/-- Acceptance binds the checked graph to the actual parsed graph, including its
ordered, possibly repeated outputs; validity comes from the checker theorem. -/
theorem decode_sound {text : String} {checked : CheckedGraph}
    (accepted : decode text = .ok checked) :
    ∃ raw, parseRaw text = .ok raw ∧ checked.raw = raw ∧ WellTyped raw := by
  unfold decode at accepted
  cases parsed : parseRaw text with
  | error error => simp [parsed, bind, Except.bind] at accepted
  | ok raw =>
    have checkedOk : check raw = .ok checked := by
      simpa [parsed, bind, Except.bind] using accepted
    exact ⟨raw, rfl, check_sound checkedOk⟩

theorem decode_preserves_outputs {text : String} {raw : RawGraph} {checked : CheckedGraph}
    (parsed : parseRaw text = .ok raw) (accepted : decode text = .ok checked) :
    checked.raw.outputs = raw.outputs := by
  have checkedOk : check raw = .ok checked := by
    simpa [decode, parsed, bind, Except.bind] using accepted
  exact congrArg RawGraph.outputs (check_sound checkedOk).1

/-- Byte acceptance also records the exact successfully decoded UTF-8 text. -/
theorem decodeBytes_sound {bytes : ByteArray} {checked : CheckedGraph}
    (accepted : decodeBytes bytes = .ok checked) :
    ∃ text raw, String.fromUTF8? bytes = some text ∧ parseRaw text = .ok raw ∧
      checked.raw = raw ∧ WellTyped raw := by
  unfold decodeBytes at accepted
  cases utf8 : String.fromUTF8? bytes with
  | none => simp [utf8] at accepted
  | some text =>
    have decoded : decode text = .ok checked := by simpa [utf8] using accepted
    obtain ⟨raw, parsed, same, valid⟩ := decode_sound decoded
    exact ⟨text, raw, rfl, parsed, same, valid⟩

/-- For every matching input, evaluation of an accepted file follows the raw
node semantics and selects the actual parsed outputs in their original order. -/
theorem decode_eval_semantics {text : String} {checked : CheckedGraph}
    (accepted : decode text = .ok checked) (inputs : Inputs)
    (matching : InputsMatch checked.raw inputs) :
    ∃ raw, parseRaw text = .ok raw ∧ checked.raw = raw ∧ WellTyped raw ∧
      EvalNodes inputs [] raw.nodes (evalValues checked inputs matching).buffers ∧
      OutputSelection (evalValues checked inputs matching).buffers raw.outputs
        (eval checked inputs matching) := by
  obtain ⟨raw, parsed, same, _⟩ := decode_sound accepted
  have parsedChecked : parseRaw text = .ok checked.raw := by simpa [← same] using parsed
  exact ⟨checked.raw, parsedChecked, rfl, checked.wellTyped,
    eval_semantics checked inputs matching⟩

theorem decodeBytes_eval_semantics {bytes : ByteArray} {checked : CheckedGraph}
    (accepted : decodeBytes bytes = .ok checked) (inputs : Inputs)
    (matching : InputsMatch checked.raw inputs) :
    ∃ text raw, String.fromUTF8? bytes = some text ∧ parseRaw text = .ok raw ∧
      checked.raw = raw ∧ WellTyped raw ∧
      EvalNodes inputs [] raw.nodes (evalValues checked inputs matching).buffers ∧
      OutputSelection (evalValues checked inputs matching).buffers raw.outputs
        (eval checked inputs matching) := by
  obtain ⟨text, raw, utf8, parsed, same, _⟩ := decodeBytes_sound accepted
  have parsedChecked : parseRaw text = .ok checked.raw := by simpa [← same] using parsed
  exact ⟨text, checked.raw, utf8, parsedChecked, rfl, checked.wellTyped,
    eval_semantics checked inputs matching⟩

/-- Each position in the parsed output list selects the corresponding evaluated
node buffer and its logical shape, even when references repeat. -/
theorem decode_eval_output_typed {text : String} {raw : RawGraph} {checked : CheckedGraph}
    (parsed : parseRaw text = .ok raw) (accepted : decode text = .ok checked)
    (inputs : Inputs) (matching : InputsMatch checked.raw inputs) {index id : Nat}
    (position : raw.outputs[index]? = some id) :
    ∃ shape value,
      (outputShapes checked)[index]? = some shape ∧
      checked.shapes[id]? = some shape ∧
      (eval checked inputs matching)[index]? = some value ∧
      (evalValues checked inputs matching).buffers[id]? = some value ∧
      value.length = numel shape := by
  have checkedOk : check raw = .ok checked := by
    simpa [decode, parsed, bind, Except.bind] using accepted
  have same := (check_sound checkedOk).1
  apply eval_output_typed checked inputs matching
  simpa [same] using position

end VtcProofs.StrictGraph
