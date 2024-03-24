mod basic;
mod common;
mod image_compute;

use std::borrow::BorrowMut;

use anyhow::Result;
use image::{load_from_memory, DynamicImage, GenericImage, ImageBuffer, ImageFormat, Rgba};

use crate::{basic::basic, image_compute::image};

fn main() -> Result<()> {
    println!("Run basic");
    basic("saxpy_float", include_str!("./basic/basic.cl"))?;

    println!("Run image");
    let image_data = image("colorize", include_str!("./image_compute/image.cl"))?;
    let db = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, image_data).unwrap();
    db.save_with_format("image_compute.png", ImageFormat::Png)?;

    Ok(())
}
