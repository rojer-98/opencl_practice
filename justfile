default: (run)

alias r := run
alias f := flamegraph

run:
  RUSTICL_ENABLE=llvmpipe cargo run --release

flamegraph:
  RUSTICL_ENABLE=llvmpipe cargo flamegraph --release 
