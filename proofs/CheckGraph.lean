import VtcProofs.GraphEncoding

/-- Validate a source graph using the same decoder as the Lean theorems. -/
def main (args : List String) : IO UInt32 := do
  let [path] := args | do
    IO.eprintln "usage: lake env lean --run CheckGraph.lean GRAPH.json"
    return 2
  match VtcProofs.StrictGraph.decodeBytes (← IO.FS.readBinFile path) with
  | .ok _ =>
    IO.println "Well-typed strict f64 v1 source graph"
    return 0
  | .error msg =>
    IO.eprintln s!"Invalid graph: {msg}"
    return 1
