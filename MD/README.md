Usage:

  # First time setup (install dependencies)
  cd /path/to/Matlab
  julia --project -e 'using Pkg; Pkg.instantiate()'

  julia --project -e 'import Pkg; Pkg.add("CUDA")'

  julia --project -e 'import Pkg; Pkg.add("CairoMakie")'

  julia --project -e 'import Pkg; Pkg.add("ProgressMeter")' 


  # Run full simulation (~2-5 min on A6000)
  julia --project phase_boundary_gpu.jl

  # Fast test (~30 sec)
  julia --project -e 'include("phase_boundary_gpu.jl"); main(fast=true)'

  Output:
  - phase_boundary_gpu.png - PNG figure (high DPI)
  - phase_boundary_gpu.pdf - PDF for publication

  Key parameters (edit in code):
  N = 50              # Network dimension
  n_alpha = 100       # α grid points
  n_T = 100           # T grid points
  b = 2 + sqrt(2)     # LSR sharpness (≈3.41)
  n_eq = 2000         # Equilibration steps
  n_samp = 1500       # Sampling steps

  Expected performance on A6000:
  - Default config (100×100): ~2-5 minutes total
  - Fast config (40×40): ~30-60 seconds
