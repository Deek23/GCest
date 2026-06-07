// Batched Levenberg-Marquardt kernel template.
//
// Each CUDA block solves one nonlinear least-squares problem:
//
//      min_p  sum_i  rho( || r_i(p) ||^2 )
//
// where r_i is a scalar residual and rho is the Cauchy loss with scale a:
//      rho(s) = a^2 * log(1 + s / a^2)
//      rho'(s) = 1 / (1 + s / a^2)
//
// The weight  w_i = sqrt(rho'(r_i^2))  is applied to both the residual and
// the Jacobian row so the normal equations match an IRLS approximation of
// the robustified problem. This matches Ceres' default first-order Cauchy
// handling (Triggs correction is intentionally omitted for simplicity).
//
// Parameter bounds are enforced by clipping the trial point onto the box
// [lb, ub] after every accepted step.
//
// The template parameter `Curve` is a struct supplying:
//
//   struct Curve {
//     static constexpr int NP = ...;            // # parameters
//     template <typename T>                     // T = double or Jet<double, NP>
//     __device__ static T residual(double x, double y, const T* p);
//   };
//
// Inputs:
//   heights      [n_problems * n_ages]    row-major per pixel
//   ages         [n_ages]                 shared age axis (broadcast)
//   inits/ends   [n_problems]             valid age index range per pixel
//   init_params  [NP]                     common initial parameter vector
//   lb / ub      [NP]                     box bounds
//
// Outputs:
//   params_out   [n_problems * NP]
//   results_out  [n_problems]             LMResult per problem
//
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "Jet.cuh"
#include "Chol6x6.cuh"

namespace gcest {
namespace cuda {

struct LMConfig {
  int    max_iter             = 500;
  // Ceres trust-region starts at radius=1e4, equivalent to lambda ~ 1e-4.
  // Starting with too large a lambda makes early steps too conservative and
  // can let the algorithm get stuck near a poor initial point.
  double initial_lambda       = 1e-4;
  double lambda_up            = 2.0;     // Ceres uses 2.0 on bad step
  double lambda_down          = 0.333;   // Ceres uses 1/3 on good step
  double lambda_min           = 1e-16;
  double lambda_max           = 1e+16;
  double cauchy_a             = 0.5;     // Cauchy scale; loss = a^2 log(1 + s/a^2)
  double cost_rel_tol         = 1e-8;    // relative cost change to declare convergence (Ceres uses 1e-6)
  double param_abs_tol        = 1e-9;    // step norm tolerance (loose; cost_rel_tol is primary)
  int    max_consecutive_fail = 25;      // tolerate transient stalls; bigger for box-bounded problems
};

struct LMResult {
  double final_cost;
  int    iterations;
  int    converged;   // 1 if termination came from a tolerance, 0 if max_iter
};

// Block-wide double atomic on shared memory. Ada supports this natively.
__device__ inline void s_atomic_add(double* addr, double v) {
  atomicAdd(addr, v);
}

// Block-wide reduction of a single double using shared memory.
// `scratch` must point to at least blockDim.x doubles (or be implemented via warps).
__device__ inline double block_reduce_sum(double v, double* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) scratch[tid] += scratch[tid + s];
    __syncthreads();
  }
  return scratch[0];
}

// Compute the shared-memory bytes required by the batched LM kernel for
// a given parameter count and block size. The host wrapper must launch
// the kernel with this much dynamic shared memory.
//
// Layout (in order):
//   p[NP], p_trial[NP], dx[NP], diag0[NP]                ->  4*NP doubles
//   JtJ[NP*NP], JtJ_work[NP*NP]                          ->  2*NP*NP doubles
//   Jtr[NP]                                              ->  NP doubles
//   reduce_scratch[block_dim]                            ->  block_dim doubles
template <int NP>
constexpr inline size_t batched_lm_shared_bytes(int block_dim) {
  return sizeof(double) * (5 * NP + 2 * NP * NP + block_dim);
}

// ---------------------------------------------------------------------------
// Per-block LM kernel.
//
// Launch: <<< n_problems, block_dim, batched_lm_shared_bytes<NP>(block_dim) >>>
// Recommended block_dim = 64 for residuals ~= 80 on Ada.
// ---------------------------------------------------------------------------
template <typename Curve, int NP>
__global__ void batched_lm_kernel(
    const double* __restrict__ heights,       // [n_problems * n_ages]
    const double* __restrict__ ages,          // [n_ages]
    const int*    __restrict__ inits,         // [n_problems]  inclusive start age index
    const int*    __restrict__ ends,          // [n_problems]  inclusive end age index
    const double* __restrict__ init_params,   // [NP]
    const double* __restrict__ lb,            // [NP]
    const double* __restrict__ ub,            // [NP]
    int    n_ages,
    int    n_problems,
    LMConfig cfg,
    double* params_out,                       // [n_problems * NP]
    LMResult* results_out                     // [n_problems]
) {
  const int pix = blockIdx.x;
  if (pix >= n_problems) return;

  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // --- Shared memory layout ---
  extern __shared__ double smem[];
  double* p        = smem;                            // [NP]
  double* p_trial  = p + NP;                          // [NP]
  double* dx       = p_trial + NP;                    // [NP]
  double* diag0    = dx + NP;                         // [NP]   diag of J^T J for Marquardt damping
  double* JtJ      = diag0 + NP;                      // [NP*NP]
  double* JtJ_work = JtJ + NP * NP;                   // [NP*NP] damped copy / factor
  double* Jtr      = JtJ_work + NP * NP;              // [NP]
  double* scratch  = Jtr + NP;                        // [blk]

  // --- Per-pixel constants ---
  const int init = inits[pix];
  const int end  = ends[pix];
  const double* my_heights = heights + pix * n_ages;

  // Bounds & init params (single-thread copies are fine: NP <= blk)
  if (tid < NP) {
    p[tid] = init_params[tid];
  }
  __syncthreads();

  double lambda = cfg.initial_lambda;
  double prev_cost = 1e300;
  int    consec_fail = 0;
  int    iter_done = 0;
  int    converged = 0;
  double final_cost = 0.0;

  for (int iter = 0; iter < cfg.max_iter; ++iter) {
    iter_done = iter + 1;

    // ------------------------------------------------------------------
    // 1. Build J^T J, J^T r, and current cost via forward-mode AD.
    // ------------------------------------------------------------------
    if (tid < NP * NP) JtJ[tid] = 0.0;
    if (tid < NP)      Jtr[tid] = 0.0;
    __syncthreads();

    const double a_cauchy = cfg.cauchy_a;
    const double a2       = a_cauchy * a_cauchy;
    const double inv_a2   = 1.0 / a2;
    const double half_a2  = 0.5 * a2;

    double cost_local = 0.0;

    // Each thread covers a strided subset of the age range.
    for (int i = init + tid; i <= end; i += blk) {
      const double x = ages[i];
      const double y = my_heights[i];

      // Build Jet parameters for this evaluation.
      Jet<double, NP> p_jet[NP];
#pragma unroll
      for (int k = 0; k < NP; ++k) {
        p_jet[k] = Jet<double, NP>(p[k], k);
      }
      Jet<double, NP> r_jet = Curve::template residual<Jet<double, NP>>(x, y, p_jet);

      const double r    = r_jet.a;
      const double s    = r * r;
      // Cauchy:  rho(s)  =  a^2 * log(1 + s/a^2)
      //          rho'(s) =  1 / (1 + s/a^2)
      // IRLS Gauss-Newton step (no Triggs correction):
      //   weighted residual r_w  = sqrt(rho') * r
      //   weighted Jacobian J_w  = sqrt(rho') * J
      //   normal eqs: (J^T W J + lambda diag) dx = J^T W r
      // This drops Triggs' second-order term; convergence is slightly slower
      // than Ceres but numerically stable. Implementing correct Triggs requires
      // scaling BOTH J and r consistently (J*(1-alpha), r/(1-alpha)); the
      // earlier asymmetric version inflated the gradient and broke convergence.
      const double rho1 = 1.0 / (1.0 + s * inv_a2);
      const double w    = ::sqrt(rho1);
      const double rw   = r * w;

      double Jw[NP];
#pragma unroll
      for (int k = 0; k < NP; ++k) Jw[k] = r_jet.v[k] * w;

      // Actual Cauchy loss contribution (matches Ceres' final_cost convention):
      //   cost_i = 0.5 * rho(r_i^2) = 0.5 * a^2 * log(1 + r^2 / a^2)
      cost_local += half_a2 * ::log(1.0 + s * inv_a2);

      // Accumulate J^T J (lower triangle) and J^T r.
#pragma unroll
      for (int a = 0; a < NP; ++a) {
        s_atomic_add(&Jtr[a], Jw[a] * rw);
#pragma unroll
        for (int b = 0; b <= a; ++b) {
          s_atomic_add(&JtJ[a * NP + b], Jw[a] * Jw[b]);
        }
      }
    }
    const double cost = block_reduce_sum(cost_local, scratch);

    // ------------------------------------------------------------------
    // 2. Convergence check on cost.
    // ------------------------------------------------------------------
    if (iter > 0) {
      const double rel = ::fabs(prev_cost - cost) /
                         (cfg.cost_rel_tol + ::fabs(prev_cost));
      if (rel < cfg.cost_rel_tol) {
        converged = 1;
        final_cost = cost;
        break;
      }
    }

    // Snapshot the diagonal of J^T J once (used by Marquardt damping).
    if (tid < NP) diag0[tid] = JtJ[tid * NP + tid];
    __syncthreads();

    // ------------------------------------------------------------------
    // 3. Solve (J^T J + lambda * diag) dx = J^T r with retries on damping
    //    and a 1-pass active-set step for box constraints.
    //
    // Active-set rationale: simple clipping after an unconstrained step
    // (p_trial = clip(p - dx, lb, ub)) gives wrong directions for free
    // coords whenever a different coord wants to push outward at a binding
    // bound. Instead, we detect bound-active coords up-front and force
    // dx[k] = 0 for them by zeroing row/col k of JtJ_work and Jtr[k] before
    // factoring. This solves the reduced problem on the free set in one
    // Cholesky call. The detection uses the unconstrained sign of -J^T r
    // (the steepest-descent direction), which is exact for a one-step
    // active-set decision.
    //
    // Worst case: a coord is wrongly forced inactive — we'll just clip it
    // later. Net result is no worse than the previous pure-clip behaviour.
    // ------------------------------------------------------------------

    // Identify the active set from the steepest-descent direction (no solve yet).
    // active[k] = 1 means coord k is fixed at its bound for this iteration.
    __shared__ int active_set[NP];
    if (tid < NP) {
      const double g_k = Jtr[tid];           // gradient (LM solves with +J^T r)
      const bool at_lb = (p[tid] <= lb[tid] + 1e-12) && (g_k > 0.0); // step would decrease p
      const bool at_ub = (p[tid] >= ub[tid] - 1e-12) && (g_k < 0.0); // step would increase p
      active_set[tid] = (at_lb || at_ub) ? 1 : 0;
    }
    __syncthreads();

    bool step_ok = false;
    int  inner_fail = 0;
    while (!step_ok && inner_fail < 6) {
      // Copy lower triangle into JtJ_work.
      if (tid < NP * NP) JtJ_work[tid] = JtJ[tid];
      __syncthreads();

      if (tid == 0) {
        apply_lm_damping<NP>(JtJ_work, diag0, lambda);

        // Project out active coords: for each k in active set, set
        //   JtJ_work[k, *] = JtJ_work[*, k] = 0,  JtJ_work[k, k] = 1,
        //   Jtr[k] = 0   →  dx[k] = 0 from the solve.
        // (We operate on the LOWER triangle only since Cholesky reads only that.)
        for (int k = 0; k < NP; ++k) {
          if (active_set[k]) {
            for (int j = 0; j < NP; ++j) {
              if (j <= k) JtJ_work[k * NP + j] = (j == k) ? 1.0 : 0.0;
              else        JtJ_work[j * NP + k] = 0.0;  // below-diag of col k
            }
          }
        }
        // Use a scratch copy of Jtr for the solve so the next iteration sees
        // the original (we use Jtr again for the next active-set check).
        double Jtr_solve[NP];
        for (int k = 0; k < NP; ++k) {
          Jtr_solve[k] = active_set[k] ? 0.0 : Jtr[k];
        }

        const bool ok = cholesky_inplace<NP>(JtJ_work);
        if (ok) {
          cholesky_solve<NP>(JtJ_work, Jtr_solve, dx);
        } else {
          dx[0] = ::nan("");
        }
      }
      __syncthreads();

      const bool ok = !::isnan(dx[0]);
      if (!ok) {
        if (tid == 0) {
          lambda = ::fmin(lambda * cfg.lambda_up * cfg.lambda_up, cfg.lambda_max);
        }
        __syncthreads();
        ++inner_fail;
        continue;
      }
      step_ok = true;
    }

    if (!step_ok) {
      final_cost = cost;
      converged = 0;
      break;
    }

    // ------------------------------------------------------------------
    // 4. Trial step p_trial = clip(p - dx, lb, ub) and evaluate new cost.
    //    The clip is now mostly a safety net (active coords have dx=0;
    //    free coords might still overshoot a bound, in which case clip
    //    handles it).
    // ------------------------------------------------------------------
    if (tid < NP) {
      double np_k = p[tid] - dx[tid];
      np_k = ::fmin(::fmax(np_k, lb[tid]), ub[tid]);
      p_trial[tid] = np_k;
    }
    __syncthreads();

    // Compute trial cost (no Jacobian needed) using the SAME loss formula
    // as the accumulation phase, so accept/reject is consistent.
    double trial_local = 0.0;
    for (int i = init + tid; i <= end; i += blk) {
      const double x = ages[i];
      const double y = my_heights[i];
      const double r = Curve::template residual<double>(x, y, p_trial);
      const double s = r * r;
      trial_local += half_a2 * ::log(1.0 + s * inv_a2);
    }
    const double trial_cost = block_reduce_sum(trial_local, scratch);

    // ------------------------------------------------------------------
    // 5. Accept / reject.
    // ------------------------------------------------------------------
    if (trial_cost < cost) {
      // Accept. Compute step norm for convergence check.
      double step_inf = 0.0;
      if (tid < NP) {
        step_inf = ::fabs(p_trial[tid] - p[tid]);
      }
      // Reduce step norm (only first NP threads carry valid contributions).
      step_inf = block_reduce_sum(tid < NP ? step_inf : 0.0, scratch);
      // Above is a SUM not a MAX, but as a coarse convergence indicator it suffices;
      // a true L-infty would require a different reduction.

      // Commit.
      if (tid < NP) p[tid] = p_trial[tid];
      __syncthreads();

      if (tid == 0) {
        lambda = ::fmax(lambda * cfg.lambda_down, cfg.lambda_min);
      }
      consec_fail = 0;
      prev_cost = cost;
      final_cost = trial_cost;

      if (step_inf < cfg.param_abs_tol) {
        converged = 1;
        break;
      }
    } else {
      // Reject. Increase damping, do NOT update p.
      if (tid == 0) {
        lambda = ::fmin(lambda * cfg.lambda_up, cfg.lambda_max);
      }
      ++consec_fail;
      final_cost = cost;
      if (consec_fail >= cfg.max_consecutive_fail) {
        // Stalled.
        converged = 0;
        break;
      }
    }
    __syncthreads();
  }

  // ----------------------------------------------------------------------
  // Write results.
  // ----------------------------------------------------------------------
  if (tid < NP) {
    params_out[pix * NP + tid] = p[tid];
  }
  if (tid == 0) {
    LMResult res;
    res.final_cost = final_cost;
    res.iterations = iter_done;
    res.converged  = converged;
    results_out[pix] = res;
  }
}

}  // namespace cuda
}  // namespace gcest
