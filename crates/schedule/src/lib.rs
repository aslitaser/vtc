//! Schedule transforms and translation validation for `vtc`.
//!
//! This crate will choose and validate loop schedule transformations. It sits
//! above `vtc-loopir` and accepts only schedules that preserve loop semantics.
