kernel void colorize(write_only image2d_t image) {
  const size_t x = get_global_id(0);
  const size_t y = get_global_id(1);
  write_imageui(image, (int2)(x, y), (uint4)(x, y, 0, 255));
}
