import VtcProofs.GraphEncoding

open VtcProofs.StrictGraph

private def expectAccepted (label text : String) : IO CheckedGraph :=
  match decode text with
  | .error error => throw <| IO.userError s!"{label}: decoder rejected graph: {error}"
  | .ok graph => pure graph

private def expectRejected (label text : String) : IO Unit :=
  match decode text with
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError s!"{label}: decoder accepted invalid graph"

private def fixtureName : String := "features\"\\\n雪"

private def fixtureRaw : RawGraph := {
  nodes := [
    .input fixtureName [2],
    .const [2] [0x8000000000000000, 0xc000000000000000],
    .add 0 1,
    .relu 2,
    .add 2 2
  ]
  outputs := [4, 3, 2, 3, 0]
}

private def expectEvaluation (label : String) (graph : CheckedGraph) (inputs : Inputs)
    (expected : List (List UInt64)) : IO Unit := do
  if matching : InputsMatch graph.raw inputs then
    let actual := (eval graph inputs matching).map (fun values => values.map (·.toBits))
    unless actual == expected do
      throw <| IO.userError s!"{label}: evaluated output bits differ: {repr actual}"
  else
    throw <| IO.userError s!"{label}: test inputs do not match the decoded graph"

private def rejectionCases : List (String × String) := [
  ("invalid JSON", "["),
  ("trailing garbage", r#"["vtc.graph",1,[["input","f64","x",[1]]],[0]] false"#),
  ("trailing array", r#"["vtc.graph",1,[["input","f64","x",[1]]],[0]][]"#),
  ("root object", r#"{"format":"vtc.graph","version":1}"#),
  ("root arity", r#"["vtc.graph",1,[["input","f64","x",[1]]],[0],0]"#),
  ("unknown format", r#"["other.graph",1,[["input","f64","x",[1]]],[0]]"#),
  ("unknown version", r#"["vtc.graph",2,[["input","f64","x",[1]]],[0]]"#),
  ("string version", r#"["vtc.graph","1",[["input","f64","x",[1]]],[0]]"#),
  ("fractional version", r#"["vtc.graph",1.5,[["input","f64","x",[1]]],[0]]"#),
  ("negative version", r#"["vtc.graph",-1,[["input","f64","x",[1]]],[0]]"#),
  ("unknown dtype", r#"["vtc.graph",1,[["input","f32","x",[1]]],[0]]"#),
  ("unknown operation", r#"["vtc.graph",1,[["input","f64","x",[1]],["neg",0]],[1]]"#),
  ("input arity", r#"["vtc.graph",1,[["input","f64","x",[1],0]],[0]]"#),
  ("input name type", r#"["vtc.graph",1,[["input","f64",1,[1]]],[0]]"#),
  ("shape type", r#"["vtc.graph",1,[["input","f64","x",1]],[0]]"#),
  ("negative dimension", r#"["vtc.graph",1,[["input","f64","x",[-1]]],[0]]"#),
  ("fractional dimension", r#"["vtc.graph",1,[["input","f64","x",[1.5]]],[0]]"#),
  ("string dimension", r#"["vtc.graph",1,[["input","f64","x",["1"]]],[0]]"#),
  ("zero dimension", r#"["vtc.graph",1,[["input","f64","x",[0]]],[0]]"#),
  ("overflowing shape product", r#"["vtc.graph",1,[["input","f64","x",[4294967296,4294967296]]],[0]]"#),
  ("add arity", r#"["vtc.graph",1,[["input","f64","x",[1]],["add",0,0,0]],[1]]"#),
  ("relu arity", r#"["vtc.graph",1,[["input","f64","x",[1]],["relu",0,0]],[1]]"#),
  ("negative node id", r#"["vtc.graph",1,[["input","f64","x",[1]],["relu",-1]],[1]]"#),
  ("fractional node id", r#"["vtc.graph",1,[["input","f64","x",[1]],["relu",0.5]],[1]]"#),
  ("string node id", r#"["vtc.graph",1,[["input","f64","x",[1]],["relu","0"]],[1]]"#),
  ("forward reference", r#"["vtc.graph",1,[["relu",1],["input","f64","x",[1]]],[0]]"#),
  ("self reference", r#"["vtc.graph",1,[["relu",0]],[0]]"#),
  ("output type", r#"["vtc.graph",1,[["input","f64","x",[1]]],["0"]]"#),
  ("negative output", r#"["vtc.graph",1,[["input","f64","x",[1]]],[-1]]"#),
  ("output out of range", r#"["vtc.graph",1,[["input","f64","x",[1]]],[1]]"#),
  ("empty outputs", r#"["vtc.graph",1,[["input","f64","x",[1]]],[]]"#),
  ("constant arity", r#"["vtc.graph",1,[["const","f64",[1],["0000000000000000"],0]],[0]]"#),
  ("constant values type", r#"["vtc.graph",1,[["const","f64",[1],"0000000000000000"]],[0]]"#),
  ("bits as number", r#"["vtc.graph",1,[["const","f64",[1],[0]]],[0]]"#),
  ("short bits", r#"["vtc.graph",1,[["const","f64",[1],["000000000000000"]]],[0]]"#),
  ("long bits", r#"["vtc.graph",1,[["const","f64",[1],["00000000000000000"]]],[0]]"#),
  ("uppercase bits", r#"["vtc.graph",1,[["const","f64",[1],["3FF0000000000000"]]],[0]]"#),
  ("prefixed bits", r#"["vtc.graph",1,[["const","f64",[1],["0x00000000000000"]]],[0]]"#),
  ("nonhex bits", r#"["vtc.graph",1,[["const","f64",[1],["000000000000000g"]]],[0]]"#),
  ("constant size mismatch", r#"["vtc.graph",1,[["const","f64",[2],["0000000000000000"]]],[0]]"#),
  ("infinite constant", r#"["vtc.graph",1,[["const","f64",[1],["7ff0000000000000"]]],[0]]"#),
  ("NaN constant", r#"["vtc.graph",1,[["const","f64",[1],["7ff8000000000001"]]],[0]]"#),
  ("add shape mismatch", r#"["vtc.graph",1,[["input","f64","x",[1]],["input","f64","y",[2]],["add",0,1]],[2]]"#),
  ("same-numel add shape mismatch", r#"["vtc.graph",1,[["input","f64","x",[2]],["input","f64","y",[1,2]],["add",0,1]],[2]]"#),
  ("duplicate input names", r#"["vtc.graph",1,[["input","f64","x",[1]],["input","f64","x",[1]]],[0]]"#)
]

def main : IO Unit := do
  let scalar ← expectAccepted "rank-zero negative zero"
    r#"["vtc.graph",1,[["const","f64",[],["8000000000000000"]]],[0,0]]"#
  unless scalar.raw.outputs == [0, 0] do
    throw <| IO.userError "rank-zero graph lost duplicate outputs"
  expectEvaluation "rank-zero duplicate negative zero" scalar (fun _ => [])
    [[0x8000000000000000], [0x8000000000000000]]
  let _ ← expectAccepted "minimum subnormal and finite maximum"
    r#"["vtc.graph",1,[["const","f64",[2],["0000000000000001","7fefffffffffffff"]]],[0]]"#
  let _ ← expectAccepted "surrounding whitespace"
    " \n\t[\"vtc.graph\",1,[[\"input\",\"f64\",\"x\",[1]]],[0]] \r\n"
  let mixedRanks ← expectAccepted "ordered mixed-rank outputs"
    r#"["vtc.graph",1,[["input","f64","vector",[1]],["const","f64",[],["3ff0000000000000"]]],[1,0,1]]"#
  unless outputShapes mixedRanks == [[], [1], []] do
    throw <| IO.userError "mixed-rank output shapes lost order or duplicate references"
  expectEvaluation "ordered mixed-rank output values" mixedRanks
    (fun _ => [VtcProofs.StrictF64.decode 0x4000000000000000])
    [[0x3ff0000000000000], [0x4000000000000000], [0x3ff0000000000000]]
  for (label, text) in rejectionCases do
    expectRejected label text
  match decodeBytes (ByteArray.mk #[0xff]) with
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError "invalid UTF-8 was accepted"
  let fixture ← IO.FS.readBinFile "fixtures/strict-graph-v1.json"
  let graph ← match decodeBytes fixture with
    | .error error => throw <| IO.userError s!"shared Rust fixture was rejected: {error}"
    | .ok graph => pure graph
  unless graph.raw == fixtureRaw do
    throw <| IO.userError "shared fixture raw nodes, name, bits, sharing or ordered outputs changed"
  let inputs : Inputs := fun name =>
    if name == fixtureName then
      [VtcProofs.StrictF64.decode 0x8000000000000000,
        VtcProofs.StrictF64.decode 0x4008000000000000]
    else []
  for length in [0, 1, 3] do
    let wrongLength : Inputs := fun _ => List.replicate length VtcProofs.StrictF64.positiveZero
    if InputsMatch graph.raw wrongLength then
      throw <| IO.userError s!"runtime input length {length} matched shape [2]"
  expectEvaluation "shared Rust fixture" graph inputs [
    [0x8000000000000000, 0x4000000000000000],
    [0, 0x3ff0000000000000],
    [0x8000000000000000, 0x3ff0000000000000],
    [0, 0x3ff0000000000000],
    [0x8000000000000000, 0x4008000000000000]
  ]
  let nonfinite ← expectAccepted "nonfinite runtime inputs"
    r#"["vtc.graph",1,[["input","f64","special",[3]],["relu",0]],[1]]"#
  let specialInputs : Inputs := fun _ =>
    [VtcProofs.StrictF64.decode 0x7ff0000000000001,
      VtcProofs.StrictF64.decode 0x7ff0000000000000,
      VtcProofs.StrictF64.decode 0xfff0000000000000]
  expectEvaluation "NaN and infinities as runtime inputs" nonfinite specialInputs
    [[0, 0x7ff0000000000000, 0]]
  IO.println s!"graph decoder: {rejectionCases.length} rejection cases and fixture passed"
