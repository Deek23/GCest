// ceres::Solver::Options を CPU / CUDA で切り替える共通ヘルパ
#pragma once
#include <ceres/ceres.h>

namespace backend {

inline void configureSolver(ceres::Solver::Options& options,
                            int maxIter,
                            bool useCUDA) {
  options.max_num_iterations = maxIter;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = false;
  options.trust_region_problem_dump_directory = "./tmp";
  options.trust_region_problem_dump_format_type = ceres::TEXTFILE;
  options.use_explicit_schur_complement = true;

#ifdef GCEST_USE_CUDA
  if (useCUDA) {
    // Ceres 2.2+ では DENSE_QR + CUDA が利用可能（正定値性を要求しないため頑健）
    // 2.2 未満では DENSE_NORMAL_CHOLESKY しか CUDA 対応していないため、
    // cusolverDnDpotrf 失敗時のフォールバック対策として damping を強める。
    options.linear_solver_type = ceres::DENSE_QR;  // 要 Ceres >= 2.2
    options.dense_linear_algebra_library_type = ceres::CUDA;
    // 数値的安定性向上のための保険
    options.use_nonmonotonic_steps = true;
    options.max_num_consecutive_invalid_steps = 10;
    options.min_trust_region_radius = 1e-12;
    return;
  }
#else
  (void)useCUDA;
#endif

  // CPU 既定（従来挙動）
  options.linear_solver_type = ceres::DENSE_QR;
}

}  // namespace backend
