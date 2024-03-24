mod basic;

use anyhow::Result;

use basic::basic;

fn main() -> Result<()> {
    println!("Run basic");
    basic("saxpy_float", include_str!("./basic/basic.cl"))?;

    Ok(())
}
