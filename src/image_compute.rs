use std::ffi::c_void;

use anyhow::{anyhow, Result};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    kernel::{ExecuteKernel, Kernel},
    memory::{Image, CL_MEM_OBJECT_IMAGE2D, CL_MEM_WRITE_ONLY, CL_RGBA, CL_UNSIGNED_INT8},
    program::{Program, CL_STD_2_0},
    types::cl_event,
    types::{cl_image_desc, cl_image_format, CL_NON_BLOCKING},
};

use crate::common::get_device_context;

pub fn image(kernel_name: &str, source: &str) -> Result<Vec<u8>> {
    let (device, context) = get_device_context()?;

    // Print some information about the device
    println!(
        "CL_DEVICE_IMAGE_SUPPORT: {:?}",
        device.image_support().unwrap()
    );
    println!(
        "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {:?}",
        device.max_read_write_image_args().unwrap()
    );
    println!(
        "CL_DEVICE_MAX_READ_IMAGE_ARGS: {:?}",
        device.max_read_image_args().unwrap()
    );
    println!(
        "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: {:?}",
        device.max_write_image_args().unwrap()
    );
    println!(
        "CL_DEVICE_MAX_SAMPLERS: {:?}",
        device.max_device_samples().unwrap()
    );
    let supported_formats =
        context.get_supported_image_formats(CL_MEM_WRITE_ONLY, CL_MEM_OBJECT_IMAGE2D)?;
    if supported_formats
        .iter()
        .filter(|f| {
            f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_UNSIGNED_INT8
        })
        .count()
        <= 0
    {
        return Err(anyhow!(
            "Device does not support CL_RGBA with CL_UNSIGNED_INT8 for CL_MEM_WRITE_ONLY!"
        ));
    }

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, source, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, kernel_name).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .expect("CommandQueue::create_default_with_properties failed");

    // Create an image
    let mut image = unsafe {
        Image::create(
            &context,
            CL_MEM_WRITE_ONLY,
            &cl_image_format {
                image_channel_order: CL_RGBA,
                image_channel_data_type: CL_UNSIGNED_INT8,
            },
            &cl_image_desc {
                image_type: CL_MEM_OBJECT_IMAGE2D,
                image_width: 1024 as usize,
                image_height: 1024 as usize,
                image_depth: 1,
                image_array_size: 1,
                image_row_pitch: 0,
                image_slice_pitch: 0,
                num_mip_levels: 0,
                num_samples: 0,
                buffer: std::ptr::null_mut(),
            },
            std::ptr::null_mut(),
        )
        .expect("Image::create failed")
    };

    // Run the kernel on the input data
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&image)
            .set_global_work_sizes(&[1024usize, 1024usize])
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Fill the middle of the image with a solid color
    let fill_color = [11u32, 22u32, 33u32, 44u32];
    let fill_event = unsafe {
        queue.enqueue_fill_image(
            &mut image,
            fill_color.as_ptr() as *const c_void,
            &[3usize, 3usize, 0usize] as *const usize,
            &[4usize, 4usize, 1usize] as *const usize,
            &events,
        )?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(fill_event.get());

    // Read the image data from the device
    let mut image_data = [0u8; 1024 * 1024 * 4];
    let read_event = unsafe {
        queue.enqueue_read_image(
            &image,
            CL_NON_BLOCKING,
            &[0usize, 0usize, 0usize] as *const usize,
            &[1024usize, 1024usize, 1usize] as *const usize,
            0,
            0,
            image_data.as_mut_ptr() as *mut c_void,
            &events,
        )?
    };

    // Wait for the read_event to complete.
    read_event.wait()?;

    Ok(image_data.to_vec())
}
