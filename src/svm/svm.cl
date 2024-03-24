kernel void inclusive_scan_int(global int *output,
                               global int const *values) {
  int sum      = 0;
  size_t lid   = get_local_id(0);
  size_t lsize = get_local_size(0);

  size_t num_groups = get_num_groups(0);
  for (size_t i = 0u; i < num_groups; ++i) {
    size_t lidx  = i * lsize + lid;
    int value    = work_group_scan_inclusive_add(values[lidx]);
    output[lidx] = sum + value;

    sum += work_group_broadcast(value, lsize - 1);
  }
}
