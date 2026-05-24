#pragma once

#include <cuda_runtime.h>

struct GpuTimer {
  cudaEvent_t start{};
  cudaEvent_t stop{};

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void record_start(cudaStream_t stream) { cudaEventRecord(start, stream); }

  float record_stop_ms(cudaStream_t stream) {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};
