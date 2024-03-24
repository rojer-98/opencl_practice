use anyhow::{anyhow, Result};

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    error_codes::cl_int,
    kernel::{ExecuteKernel, Kernel},
    memory::{CL_MAP_READ, CL_MAP_WRITE},
    program::{Program, CL_STD_2_0},
    svm::SvmVec,
    types::CL_BLOCKING,
};

use crate::common::get_device_context;

pub fn svm(kernel_name: &str, source: &str) -> Result<Vec<i32>> {
    let (_, context) = get_device_context()?;
    let svm_caps = context.get_svm_mem_capability();
    println!("SVM capabilities: {svm_caps}");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, source, CL_STD_2_0)
        .map_err(|e| anyhow!("{e}"))?;
    let kernel = Kernel::create(&program, kernel_name)?;

    // Create a command_queue on the Context's device
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

    // The input data
    const ARRAY_SIZE: usize = 8;
    let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

    // Create an OpenCL SVM vector
    let mut test_values = SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE)?;

    // Map test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !test_values.is_fine_grained() {
        unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut test_values, &[])? };
    }

    // Copy input data into the OpenCL SVM vector
    test_values.clone_from_slice(&value_array);

    // Make test_values immutable
    let test_values = test_values;

    // Unmap test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !test_values.is_fine_grained() {
        let unmap_test_values_event = unsafe { queue.enqueue_svm_unmap(&test_values, &[])? };
        unmap_test_values_event.wait()?;
    }

    // The output data, an OpenCL SVM vector
    let mut results = SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE)?;

    // Run the kernel on the input data
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg_svm(results.as_mut_ptr())
            .set_arg_svm(test_values.as_ptr())
            .set_global_work_size(ARRAY_SIZE)
            .enqueue_nd_range(&queue)?
    };

    // Wait for the kernel to complete execution on the device
    kernel_event.wait()?;

    // Map results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !results.is_fine_grained() {
        unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut results, &[])? };
    }

    // Unmap results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !results.is_fine_grained() {
        let unmap_results_event = unsafe { queue.enqueue_svm_unmap(&results, &[])? };
        unmap_results_event.wait()?;
    }

    Ok(results.to_vec())
}
