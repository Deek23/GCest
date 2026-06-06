// Small dense SPD solvers for the per-pixel normal equation step inside LM.
// Templated on N so the same code serves Mitscherlich/Gompertz/Richards/Logistic (N=3)
// and ONRH (N=6). Designed for full unrolling on the GPU.
//
// All routines take a row-major lower-triangular layout:  A[i*N + j], j <= i.
// Only the lower triangle is read/written so the caller may leave the upper
// triangle uninitialised.
//
// Both LLT (Cholesky) and a damped-LDLT-style regularization helper are provided.
// LLT returns false when the matrix is not SPD — the caller must then increase
// the LM damping and retry.
#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace gcest {
namespace cuda {

// In-place Cholesky (A = L L^T). Lower triangle of A is overwritten with L.
// Returns false if A is not positive definite.
template <int N>
__host__ __device__ inline bool cholesky_inplace(double* A) {
#pragma unroll
  for (int j = 0; j < N; ++j) {
    double diag = A[j * N + j];
#pragma unroll
    for (int k = 0; k < j; ++k) {
      diag -= A[j * N + k] * A[j * N + k];
    }
    if (!(diag > 0.0)) return false;  // catches NaN, Inf, and <=0
    const double l_jj = ::sqrt(diag);
    A[j * N + j] = l_jj;
    const double inv = 1.0 / l_jj;

#pragma unroll
    for (int i = j + 1; i < N; ++i) {
      double s = A[i * N + j];
#pragma unroll
      for (int k = 0; k < j; ++k) {
        s -= A[i * N + k] * A[j * N + k];
      }
      A[i * N + j] = s * inv;
    }
  }
  return true;
}

// Solve L L^T x = b given the factor stored in lower triangle of A.
// x and b may alias.
template <int N>
__host__ __device__ inline void cholesky_solve(const double* A, const double* b, double* x) {
  double y[N];

  // Forward substitution: L y = b
#pragma unroll
  for (int i = 0; i < N; ++i) {
    double s = b[i];
#pragma unroll
    for (int k = 0; k < i; ++k) {
      s -= A[i * N + k] * y[k];
    }
    y[i] = s / A[i * N + i];
  }
  // Backward substitution: L^T x = y
#pragma unroll
  for (int i = N - 1; i >= 0; --i) {
    double s = y[i];
#pragma unroll
    for (int k = i + 1; k < N; ++k) {
      s -= A[k * N + i] * x[k];
    }
    x[i] = s / A[i * N + i];
  }
}

// Apply Levenberg-Marquardt damping in-place.
// A_damped[i,i] += lambda * diag_orig[i]    (Marquardt-style scaling)
// `diag_orig` must hold the diagonal of the original (un-damped) J^T J.
template <int N>
__host__ __device__ inline void apply_lm_damping(double* A, const double* diag_orig, double lambda) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    A[i * N + i] += lambda * diag_orig[i];
  }
}

// Copy lower triangle of `src` to `dst` (both row-major, dimension N).
template <int N>
__host__ __device__ inline void copy_lower(const double* src, double* dst) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j <= i; ++j) {
      dst[i * N + j] = src[i * N + j];
    }
  }
}

}  // namespace cuda
}  // namespace gcest
