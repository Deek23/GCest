// Forward-mode AD Jet type for CUDA device code.
// Mirrors ceres::Jet semantics so that residual functions written for
// AutoDiffCostFunction can be reused on the GPU with minimal changes.
//
// A Jet<T, N> carries (value, [d/dp_0, d/dp_1, ..., d/dp_{N-1}]).
// Initialize with Jet<T,N>(value, k) to set the k-th derivative to 1.
//
// All operators are __host__ __device__ so the same residual can be
// compiled for both CPU verification and GPU execution.
#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace gcest {
namespace cuda {

template <typename T, int N>
struct Jet {
  T a;        // scalar value
  T v[N];     // derivative vector

  __host__ __device__ Jet() : a(T(0)) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = T(0);
  }

  __host__ __device__ explicit Jet(const T& value) : a(value) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = T(0);
  }

  __host__ __device__ Jet(const T& value, int k) : a(value) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = (i == k) ? T(1) : T(0);
  }
};

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator+(const Jet<T, N>& f) { return f; }

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator-(const Jet<T, N>& f) {
  Jet<T, N> r;
  r.a = -f.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = -f.v[i];
  return r;
}

// ---------------------------------------------------------------------------
// Jet + Jet
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator+(const Jet<T, N>& f, const Jet<T, N>& g) {
  Jet<T, N> r;
  r.a = f.a + g.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = f.v[i] + g.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator-(const Jet<T, N>& f, const Jet<T, N>& g) {
  Jet<T, N> r;
  r.a = f.a - g.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = f.v[i] - g.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator*(const Jet<T, N>& f, const Jet<T, N>& g) {
  Jet<T, N> r;
  r.a = f.a * g.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = f.a * g.v[i] + f.v[i] * g.a;
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator/(const Jet<T, N>& f, const Jet<T, N>& g) {
  Jet<T, N> r;
  const T inv_g = T(1) / g.a;
  r.a = f.a * inv_g;
  const T inv_g2 = inv_g * inv_g;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    r.v[i] = (f.v[i] * g.a - f.a * g.v[i]) * inv_g2;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Jet + scalar
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator+(const Jet<T, N>& f, T s) {
  Jet<T, N> r = f;
  r.a = f.a + s;
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator+(T s, const Jet<T, N>& f) { return f + s; }

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator-(const Jet<T, N>& f, T s) {
  Jet<T, N> r = f;
  r.a = f.a - s;
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator-(T s, const Jet<T, N>& f) {
  Jet<T, N> r;
  r.a = s - f.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = -f.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator*(const Jet<T, N>& f, T s) {
  Jet<T, N> r;
  r.a = f.a * s;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = f.v[i] * s;
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator*(T s, const Jet<T, N>& f) { return f * s; }

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator/(const Jet<T, N>& f, T s) {
  Jet<T, N> r;
  const T inv_s = T(1) / s;
  r.a = f.a * inv_s;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = f.v[i] * inv_s;
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> operator/(T s, const Jet<T, N>& f) {
  Jet<T, N> r;
  const T inv_f = T(1) / f.a;
  r.a = s * inv_f;
  const T factor = -s * inv_f * inv_f;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = factor * f.v[i];
  return r;
}

// ---------------------------------------------------------------------------
// Compound assignment (handy for accumulators)
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline Jet<T, N>& operator+=(Jet<T, N>& f, const Jet<T, N>& g) {
  f.a += g.a;
#pragma unroll
  for (int i = 0; i < N; ++i) f.v[i] += g.v[i];
  return f;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N>& operator-=(Jet<T, N>& f, const Jet<T, N>& g) {
  f.a -= g.a;
#pragma unroll
  for (int i = 0; i < N; ++i) f.v[i] -= g.v[i];
  return f;
}

// ---------------------------------------------------------------------------
// Comparisons — only the value part matters (matches Ceres semantics).
// Used by branches such as (x < c) in the residual.
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline bool operator<(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a <  g.a; }
template <typename T, int N>
__host__ __device__ inline bool operator<(const Jet<T, N>& f, T s) { return f.a <  s; }
template <typename T, int N>
__host__ __device__ inline bool operator<(T s, const Jet<T, N>& f) { return s   <  f.a; }
template <typename T, int N>
__host__ __device__ inline bool operator>(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a >  g.a; }
template <typename T, int N>
__host__ __device__ inline bool operator>(const Jet<T, N>& f, T s) { return f.a >  s; }
template <typename T, int N>
__host__ __device__ inline bool operator>(T s, const Jet<T, N>& f) { return s   >  f.a; }
template <typename T, int N>
__host__ __device__ inline bool operator<=(const Jet<T, N>& f, T s) { return f.a <= s; }
template <typename T, int N>
__host__ __device__ inline bool operator>=(const Jet<T, N>& f, T s) { return f.a >= s; }
template <typename T, int N>
__host__ __device__ inline bool operator==(const Jet<T, N>& f, T s) { return f.a == s; }

// ---------------------------------------------------------------------------
// Scalar overloads.
// The growth-curve residuals are templated on T = {double, Jet<double, N>}.
// Inside namespace gcest::cuda, unqualified lookup of sqrt/pow/exp/log for
// a `double` argument does not fall back to ::std::sqrt etc. through ADL
// (a `double` has no associated namespace). These thin forwarders make the
// same source compile cleanly for both T = double and T = Jet.
// ---------------------------------------------------------------------------
__host__ __device__ inline double sqrt(double x) { return ::sqrt(x); }
__host__ __device__ inline double pow (double x, double y) { return ::pow(x, y); }
__host__ __device__ inline double exp (double x) { return ::exp(x); }
__host__ __device__ inline double log (double x) { return ::log(x); }
__host__ __device__ inline double abs (double x) { return ::fabs(x); }

__host__ __device__ inline float  sqrt(float x) { return ::sqrtf(x); }
__host__ __device__ inline float  pow (float x, float y) { return ::powf(x, y); }
__host__ __device__ inline float  exp (float x) { return ::expf(x); }
__host__ __device__ inline float  log (float x) { return ::logf(x); }
__host__ __device__ inline float  abs (float x) { return ::fabsf(x); }

// ---------------------------------------------------------------------------
// Elementary functions used by the growth-curve residuals (Jet versions)
// ---------------------------------------------------------------------------
template <typename T, int N>
__host__ __device__ inline Jet<T, N> sqrt(const Jet<T, N>& f) {
  Jet<T, N> r;
  // Guard against sqrt of zero/negative which would yield NaN/Inf derivatives.
  const T a = (f.a > T(1e-300)) ? f.a : T(1e-300);
  const T s = ::sqrt(a);
  r.a = s;
  const T factor = T(0.5) / s;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = factor * f.v[i];
  return r;
}

// pow with constant double exponent.
// d/dx [f^n] = n * f^(n-1) * f'
template <typename T, int N>
__host__ __device__ inline Jet<T, N> pow(const Jet<T, N>& f, T exponent) {
  Jet<T, N> r;
  const T fn1 = ::pow(f.a, exponent - T(1));
  r.a = fn1 * f.a;
  const T factor = exponent * fn1;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = factor * f.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> exp(const Jet<T, N>& f) {
  Jet<T, N> r;
  const T e = ::exp(f.a);
  r.a = e;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = e * f.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> log(const Jet<T, N>& f) {
  Jet<T, N> r;
  r.a = ::log(f.a);
  const T inv_a = T(1) / f.a;
#pragma unroll
  for (int i = 0; i < N; ++i) r.v[i] = inv_a * f.v[i];
  return r;
}

template <typename T, int N>
__host__ __device__ inline Jet<T, N> abs(const Jet<T, N>& f) {
  if (f.a >= T(0)) return f;
  return -f;
}

// ---------------------------------------------------------------------------
// Trait helpers (so generic code can ask: is this a Jet?)
// ---------------------------------------------------------------------------
template <typename T>
struct IsJet { static constexpr bool value = false; };

template <typename T, int N>
struct IsJet<Jet<T, N>> { static constexpr bool value = true; };

}  // namespace cuda
}  // namespace gcest
