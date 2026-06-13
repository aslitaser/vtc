//! Code generation backend for `vtc`.
//!
//! This crate will lower validated loop IR into target code. It sits above the
//! loop-IR boundary, where later verification claims stop and backend trust begins.
