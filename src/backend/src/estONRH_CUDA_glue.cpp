// Host-side glue between Armadillo-typed InBox/OutBox and the pure
// double*/int* CUDA launcher (estONRH_CUDA.cu). Kept in a .cpp so that
// Armadillo / Rcpp can be included without poisoning the .cu compilation.
//
// IMPORTANT — include order: RcppArmadillo.h MUST come before any header
// that pulls in <armadillo> directly. RcppArmadillo predefines macros
// (notably ARMA_64BIT_WORD) that change arma::Mat's internal layout. If
// <armadillo> is included first, the resulting arma::uword is 32-bit and
// arma::Mat's field offsets shift, causing ABI mismatch with TUs that
// included RcppArmadillo first (e.g. estONRH.cpp via GCest.hpp).
//
// When WITH_CUDA=OFF, the .cu file is excluded from the build, so we
// stub out solveBatchedONRH_CUDA() to keep the symbol resolved.
#include "RcppArmadillo.h"
#include "backend/BatchedSolver.hpp"

#include <vector>
#include <cstdint>
#include <cstring>

#ifndef GCEST_USE_CUDA

namespace backend {
bool solveBatchedONRH_CUDA(const InBox&, OutBox&) {
  // CUDA support not compiled in.
  return false;
}
}  // namespace backend

#else  // GCEST_USE_CUDA

// Forward declaration of the launcher implemented in estONRH_CUDA.cu.
// extern "C" so the symbol is unmangled and links across nvcc/g++.
extern "C" bool launch_batched_onrh(
    const double* h_heights,
    int           n_ages,
    int           n_problems,
    const double* h_ages,
    const int*    h_inits,
    const int*    h_ends,
    const double* h_init_params,
    const double* h_lb,
    const double* h_ub,
    int           max_iter,
    double*       h_params_out,
    double*       h_costs_out,
    int*          h_iters_out,
    int*          h_converged_out);

namespace backend {

namespace {

// Per-pixel valid data range, replicating NpHGdata.hpp:42-52 logic.
// Output indices are into the [0, n_ages-1] age axis (NOT age values).
// The range matches the CPU loop: for (i = data.init-20; i <= data.end-20; ++i),
// which after substitution equals (i_first_change - 1) .. (i_last_change + 1).
void computePerPixelRange(const arma::mat& Data,
                          std::vector<int>& inits,
                          std::vector<int>& ends) {
  const int n_ages     = static_cast<int>(Data.n_rows);
  const int n_problems = static_cast<int>(Data.n_cols);
  inits.resize(n_problems);
  ends.resize(n_problems);

  // NpHGdata uses years = n_elem - 1 and scans j from (years-1).
  // For n_ages = 81 this means j starts at 79, NOT 80. We replicate that.
  const int j_top = n_ages - 2;

  for (int p = 0; p < n_problems; ++p) {
    const double first = Data(0, p);
    const double last  = Data(j_top, p);

    int i = 0;
    while (i < n_ages - 1 && Data(i, p) == first) ++i;

    int j = j_top;
    while (j > 0 && Data(j, p) == last) --j;

    // CPU loop bounds:  for k = (i+19)-20 .. (j+21)-20
    //                       = (i-1) .. (j+1)
    int idx_init = i - 1;
    int idx_end  = j + 1;

    // Clip to valid index range (GPU is strict about OOB; CPU silently read OOB).
    if (idx_init < 0)        idx_init = 0;
    if (idx_end >= n_ages)   idx_end  = n_ages - 1;
    if (idx_end < idx_init)  idx_end  = idx_init;

    inits[p] = idx_init;
    ends[p]  = idx_end;
  }
}

}  // namespace

bool solveBatchedONRH_CUDA(const InBox& input_data, OutBox& outbox) {
  constexpr int NP = 6;
  const int n_ages     = static_cast<int>(input_data.Data.n_rows);
  const int n_problems = static_cast<int>(input_data.Data.n_cols);

  if (n_problems <= 0) return true;
  if (input_data.initialParameter.n_elem < (arma::uword)NP) return false;
  if (input_data.xy.n_cols < (arma::uword)n_problems) return false;
  if ((int)outbox.estimated.n_cols != n_problems ||
      (int)outbox.estimated.n_rows != NP) return false;

  // ---- ages = linspace(20, 100, n_ages), matching NpHGdata default ----
  std::vector<double> h_ages(n_ages);
  if (n_ages == 1) {
    h_ages[0] = 20.0;
  } else {
    const double step = 80.0 / (n_ages - 1);
    for (int i = 0; i < n_ages; ++i) h_ages[i] = 20.0 + step * i;
  }

  // ---- per-pixel valid range ----
  std::vector<int> h_inits, h_ends;
  computePerPixelRange(input_data.Data, h_inits, h_ends);

  // ---- heights flattened to pix-major ----
  // Armadillo stores Data in column-major: Data(age, pixel). Each pixel is a
  // contiguous column. The kernel wants heights[pix * n_ages + age], so we
  // simply transpose into a flat buffer.
  std::vector<double> h_heights(size_t(n_problems) * n_ages);
  for (int p = 0; p < n_problems; ++p) {
    const double* col_ptr = input_data.Data.colptr(p);
    std::memcpy(h_heights.data() + size_t(p) * n_ages, col_ptr,
                sizeof(double) * n_ages);
  }

  // ---- initial parameters (shared across all pixels) ----
  std::vector<double> h_init_params(NP);
  for (int k = 0; k < NP; ++k) h_init_params[k] = input_data.initialParameter(k);

  // ---- ONRH parameter bounds (mirror estONRH.cpp:67-83) ----
  const std::vector<double> h_lb = { 0.5,  5.0, 10.0, 10.0, 0.10, 0.10};
  const std::vector<double> h_ub = { 4.0, 20.0, 30.0, 60.0, 0.90, 0.90};

  // ---- output buffers ----
  std::vector<double> h_params(size_t(n_problems) * NP);
  std::vector<double> h_costs(n_problems);
  std::vector<int>    h_iters(n_problems);
  std::vector<int>    h_converged(n_problems);

  // ---- dispatch ----
  const bool ok = launch_batched_onrh(
      h_heights.data(), n_ages, n_problems,
      h_ages.data(), h_inits.data(), h_ends.data(),
      h_init_params.data(), h_lb.data(), h_ub.data(),
      input_data.maxIteration,
      h_params.data(), h_costs.data(),
      h_iters.data(), h_converged.data());

  if (!ok) return false;

  // ---- write back into OutBox (assumed pre-sized by caller) ----
  for (int p = 0; p < n_problems; ++p) {
    outbox.id(p)        = p + 1;
    outbox.xy.col(p)    = input_data.xy.col(p);
    outbox.boundary(0, p) = h_inits[p] + 20;   // age value, matching CPU output
    outbox.boundary(1, p) = h_ends[p]  + 20;
    outbox.itteration(p)  = h_converged[p];
    outbox.result(p)      = h_costs[p];
    for (int k = 0; k < NP; ++k) {
      outbox.estimated(k, p) = h_params[size_t(p) * NP + k];
    }
  }

  return true;
}

}  // namespace backend

#endif  // GCEST_USE_CUDA
