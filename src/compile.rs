use std::ptr;

use anyhow::{anyhow, Result};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING},
};
use rspirv::binary::Assemble;
use rspirv::binary::Disassemble;

use crate::common::get_device_context;

pub fn compile(kernel_name: &str, source: &str) -> Result<()> {
    let (_, context) = get_device_context()?;

    // Create a command_queue on the Context's device
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

    // Build the OpenCL program source and create the kernel.
    let program =
        Program::create_and_build_from_source(&context, source, "").map_err(|e| anyhow!("{e}"))?;
    let compiled = program
        .get_binaries()?
        .into_iter()
        .map(|v| {
            v.chunks(4)
                .map(|c| -> u32 {
                    (c[3] as u32) << 24 | (c[2] as u32) << 16 | (c[1] as u32) << 8 | c[0] as u32
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(&compiled, &mut loader).unwrap();
    let module = loader.module();

    let disasm = module.disassemble();
    println!("{disasm}");

    Ok(())
}
