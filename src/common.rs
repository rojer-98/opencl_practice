use anyhow::{anyhow, Result};
use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};

pub fn get_device_context() -> Result<(Device, Context)> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("Get all devices")
        .first()
        .ok_or(anyhow!("No device found in platform"))?;
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    Ok((device, Context::from_device(&device)?))
}
