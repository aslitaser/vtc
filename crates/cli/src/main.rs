//! Command-line entry point for `vtc`.

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "vtc", version, about = "Verified tensor compiler placeholder")]
struct Cli {}

fn main() {
    let _cli = Cli::parse();
}
