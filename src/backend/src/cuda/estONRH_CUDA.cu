// CUDA kernel launcher for batched ONRH fitting.
//
// This translation unit deliberately avoids any Armadillo / R / Rcpp headers
// because nvcc + the R macro soup is fragile. The host-side glue (.cpp)
// performs all the Armadillo <-> flat-buffer conversion and calls
// launch_batched_onrh() through the extern "C" boundary defined here.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "backend/cuda/BatchedLM.cuh"
#include "backend/cuda/ResidONRH_dev.cuh"

namespace {

inline bool check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[GCest CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

// RAII wrapper to make cleanup branchless.
struct DeviceBuf {
  void* p = nullptr;
  ~DeviceBuf() { if (p) cudaFree(p); }
  bool alloc(size_t bytes, const char* what) {
    return check_cuda(cudaMalloc(&p, bytes), what);
  }
};

}  // namespace

extern "C" bool launch_batched_onrh(
    const double* h_heights,        // [n_problems * n_ages]
    int           n_ages,
    int           n_problems,
    const double* h_ages,           // [n_ages]
    const int*    h_inits,          // [n_problems]
    const int*    h_ends,           // [n_problems]
    const double* h_init_params,    // [NP]
    const double* h_lb,             // [NP]
    const double* h_ub,             // [NP]
    int           max_iter,
    double*       h_params_out,     // [n_problems * NP]
    double*       h_costs_out,      // [n_problems]
    int*          h_iters_out,      // [n_problems]
    int*          h_converged_out)  // [n_problems]
{
  using namespace gcest::cuda;
  constexpr int NP        = ONRHCurve::NP;
  constexpr int BLOCK_DIM = 64;

  if (n_problems <= 0 || n_ages <= 0) return true;

  // Each Jet<double,6> intermediate is 56 bytes; ResidONRH evaluation creates
  // ~15 of them in stack. Bump per-thread stack to be safe.
  cudaDeviceSetLimit(cudaLimitStackSize, 4096);

  // --- Device allocations (RAII auto-frees on early return) ---
  DeviceBuf d_heights, d_ages, d_inits, d_ends;
  DeviceBuf d_init_params, d_lb, d_ub;
  DeviceBuf d_params, d_results;

  if (!d_heights.alloc(sizeof(double) * size_t(n_problems) * n_ages, "alloc heights"))    return false;
  if (!d_ages.alloc(sizeof(double) * n_ages,                        "alloc ages"))        return false;
  if (!d_inits.alloc(sizeof(int) * n_problems,                      "alloc inits"))       return false;
  if (!d_ends.alloc(sizeof(int) * n_problems,                       "alloc ends"))        return false;
  if (!d_init_params.alloc(sizeof(double) * NP,                     "alloc init_params")) return false;
  if (!d_lb.alloc(sizeof(double) * NP,                              "alloc lb"))          return false;
  if (!d_ub.alloc(sizeof(double) * NP,                              "alloc ub"))          return false;
  if (!d_params.alloc(sizeof(double) * size_t(n_problems) * NP,     "alloc params"))      return false;
  if (!d_results.alloc(sizeof(LMResult) * n_problems,               "alloc results"))     return false;

  // --- Host -> Device ---
  if (!check_cuda(cudaMemcpy(d_heights.p,     h_heights,     sizeof(double) * size_t(n_problems) * n_ages, cudaMemcpyHostToDevice), "memcpy heights"))     return false;
  if (!check_cuda(cudaMemcpy(d_ages.p,        h_ages,        sizeof(double) * n_ages,                      cudaMemcpyHostToDevice), "memcpy ages"))        return false;
  if (!check_cuda(cudaMemcpy(d_inits.p,       h_inits,       sizeof(int) * n_problems,                     cudaMemcpyHostToDevice), "memcpy inits"))       return false;
  if (!check_cuda(cudaMemcpy(d_ends.p,        h_ends,        sizeof(int) * n_problems,                     cudaMemcpyHostToDevice), "memcpy ends"))        return false;
  if (!check_cuda(cudaMemcpy(d_init_params.p, h_init_params, sizeof(double) * NP,                          cudaMemcpyHostToDevice), "memcpy init_params")) return false;
  if (!check_cuda(cudaMemcpy(d_lb.p,          h_lb,          sizeof(double) * NP,                          cudaMemcpyHostToDevice), "memcpy lb"))          return false;
  if (!check_cuda(cudaMemcpy(d_ub.p,          h_ub,          sizeof(double) * NP,                          cudaMemcpyHostToDevice), "memcpy ub"))          return false;

  // --- Kernel launch ---
  LMConfig cfg;
  cfg.max_iter = max_iter;
  // (other fields use the defaults defined in BatchedLM.cuh)

  const size_t shmem = batched_lm_shared_bytes<NP>(BLOCK_DIM);
  dim3 grid(n_problems);
  dim3 block(BLOCK_DIM);

  batched_lm_kernel<ONRHCurve, NP><<<grid, block, shmem>>>(
      static_cast<const double*>(d_heights.p),
      static_cast<const double*>(d_ages.p),
      static_cast<const int*>(d_inits.p),
      static_cast<const int*>(d_ends.p),
      static_cast<const double*>(d_init_params.p),
      static_cast<const double*>(d_lb.p),
      static_cast<const double*>(d_ub.p),
      n_ages, n_problems, cfg,
      static_cast<double*>(d_params.p),
      static_cast<LMResult*>(d_results.p));

  if (!check_cuda(cudaGetLastError(),      "kernel launch")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), "device sync"))   return false;

  // --- Device -> Host ---
  if (!check_cuda(cudaMemcpy(h_params_out, d_params.p,
                             sizeof(double) * size_t(n_problems) * NP,
                             cudaMemcpyDeviceToHost),
                  "memcpy params back")) return false;

  std::vector<LMResult> h_results(n_problems);
  if (!check_cuda(cudaMemcpy(h_results.data(), d_results.p,
                             sizeof(LMResult) * n_problems,
                             cudaMemcpyDeviceToHost),
                  "memcpy results back")) return false;

  for (int i = 0; i < n_problems; ++i) {
    h_costs_out[i]     = h_results[i].final_cost;
    h_iters_out[i]     = h_results[i].iterations;
    h_converged_out[i] = h_results[i].converged;
  }

  return true;
}
