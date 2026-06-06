// Device-side ONRH (Opposing Non-Rectangular Hyperbola) residual.
//
// Numerically identical to ResidONRH.hpp:55-57. Implemented in terms of the
// templated Jet<T,N> so the same source produces:
//   - residual<double>     : plain scalar evaluation for trial-step costing
//   - residual<Jet<double,6>> : value + 6 partials in one pass for the LM normal eqs
//
// Parameter semantics (identical to the CPU version):
//   p[0] = a    (max growth rate)
//   p[1] = m_   (initial offset)
//   p[2] = c    (inflection x)
//   p[3] = n_   (final offset)
//   p[4] = p    (acceleration curvature)
//   p[5] = q    (deceleration curvature)
//
// b is derived from the other parameters and is added to both branches.
#pragma once

#include <cuda_runtime.h>
#include "Jet.cuh"

namespace gcest {
namespace cuda {

struct ONRHCurve {
  static constexpr int NP = 6;

  // Returns y - f(x; p). T is either double or Jet<double, NP>.
  template <typename T>
  __host__ __device__ static T residual(double x, double y, const T* p) {
    // ----- derived parameter b (common to both branches) -----
    // Original (CPU) expression rewritten symbolically:
    //   pow(p[0]*(-p[2]) - p[1], 2.0) = (p[0]*p[2] + p[1])^2
    //   4 * p[0] * (-p[2]) * p[1] * p[4] = -4 * p[0]*p[2]*p[1]*p[4]
    //   b = (p[0]*p[2] + p[1] - sqrt((p[0]*p[2]+p[1])^2 - 4*p[0]*p[2]*p[1]*p[4]))
    //         / (2 * p[4])
    const T act_plus_m   = p[0] * p[2] + p[1];                 // a*c + m_
    const T b_disc       = pow(act_plus_m, 2.0)                // (a*c+m_)^2
                         - 4.0 * p[0] * p[2] * p[1] * p[4];    //   - 4 a c m_ p
    const T b            = (act_plus_m - sqrt(b_disc)) / (2.0 * p[4]);

    // ----- piecewise branch -----
    // Comparison uses only the scalar part (Jet operator<).
    if (x < p[2]) {
      // x < c   : left branch using p[1]=m_ and p[4]=p (acceleration side)
      //   inner = a*(x-c) - m_
      //   f_left = (inner + sqrt(inner^2 + 4*a*(x-c)*m_*p)) / (2*p) + b
      const T xmc        = x - p[2];                           // x - c (negative)
      const T a_xmc      = p[0] * xmc;
      const T inner      = a_xmc - p[1];
      const T disc       = pow(inner, 2.0) + 4.0 * a_xmc * p[1] * p[4];
      const T f_left     = (inner + sqrt(disc)) / (2.0 * p[4]) + b;
      return y - f_left;
    } else {
      // x >= c  : right branch using p[3]=n_ and p[5]=q (deceleration side)
      //   inner = a*(x-c) + n_
      //   f_right = (inner - sqrt(inner^2 - 4*a*(x-c)*n_*q)) / (2*q) + b
      const T xmc        = x - p[2];                           // x - c (>=0)
      const T a_xmc      = p[0] * xmc;
      const T inner      = a_xmc + p[3];
      const T disc       = pow(inner, 2.0) - 4.0 * a_xmc * p[3] * p[5];
      const T f_right    = (inner - sqrt(disc)) / (2.0 * p[5]) + b;
      return y - f_right;
    }
  }
};

}  // namespace cuda
}  // namespace gcest
