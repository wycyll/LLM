#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

inline double gemm_flops(int M, int N, int K) {
  return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
         static_cast<double>(K);
}

inline double tflops_from_ms(int M, int N, int K, double ms) {
  if (ms <= 0.0) {
    return 0.0;
  }
  return gemm_flops(M, N, K) / (ms * 1.0e-3) / 1.0e12;
}

inline double relative_error(double value, double reference) {
  const double denom = std::max(std::abs(reference), 1.0e-6);
  return std::abs(value - reference) / denom;
}
