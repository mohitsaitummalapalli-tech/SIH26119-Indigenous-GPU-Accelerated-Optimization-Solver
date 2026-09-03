# Backend Layer (`src/backend/`)

## Purpose
The Backend layer provides the hardware execution abstraction layer, isolating performance-critical execution backends (Multicore CPU and CUDA GPU) behind uniform compute contracts.

## Architecture
```
src/backend/
├── README.md       # Backend architecture overview & compute interface contracts
├── cpu/            # Multicore CPU backend (thread pools, vectorization)
│   └── README.md
└── cuda/           # CUDA GPU acceleration backend (device memory, kernels)
    └── README.md
```

## Planned Responsibilities
- **Unified Compute Contract**:
  - Abstract vector operations (`axpy`, dot product, norms).
  - Sparse Matrix-Vector Multiplication (SpMV).
  - Parallel linear system solves and triangular solves.
- **Dynamic Hardware Dispatch**:
  - Transparent routing between CPU and GPU based on problem sparsity, matrix dimensions, and hardware availability.
