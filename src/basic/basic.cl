kernel void saxpy_float(global float *z,
                        global float const *x,
                        global float const *y,
                        float a) {
  const size_t i = get_global_id(0);
  z[i]           = a * x[i] + y[i];
}
