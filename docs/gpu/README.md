# GPU Acceleration Architecture (CUDA)

This directory documents the hardware acceleration design for CUDA-enabled GPUs:

- **Hardware Acceleration Scope**:
  - Identifying bottlenecks suitable for fine-grained parallelization (Sparse Matrix-Vector multiplication, dense vector BLAS-1 operations, iterative solves).
  - Isolating algorithms from hardware specifics via the unified `src/backend/` interface.
- **Memory Management**:
  - Pinned host memory (page-locked) for overlapping PCIe data transfers and computation via CUDA streams.
  - Unified memory pools to avoid frequent dynamic allocations on the GPU device.
- **Kernel Strategy**:
  - Coalesced memory access for sparse matrix operations.
  - Warp-level reduction primitives for inner products and norm calculations.
  - Fallback criteria: small instances where host-to-device transfer latency dominates are routed to the multicore CPU backend.
