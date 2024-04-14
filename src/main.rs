mod basic;
mod common;
mod compile;
mod image_compute;
mod svm;

use anyhow::Result;
use image::{ImageBuffer, ImageFormat, Rgba};

use crate::{basic::basic, compile::compile, image_compute::image, svm::svm};

fn main() -> Result<()> {
    println!("Run basic decompile");
    compile("saxpy_float", include_str!("./basic/basic.cl"))?;

    println!("Run basic");
    basic("saxpy_float", include_str!("./basic/basic.cl"))?;

    println!("Run image");
    let image_data = image("colorize", include_str!("./image_compute/image.cl"))?;
    let db = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, image_data).unwrap();
    db.save_with_format("image_compute.png", ImageFormat::Png)?;

    println!("Run svm");
    let svm_vec = svm("inclusive_scan_int", include_str!("./svm/svm.cl"))?;
    println!("Result svm: {svm_vec:?}");

    Ok(())
}
