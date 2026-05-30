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
    // CUDA dense linear algebra: DENSE_NORMAL_CHOLESKY + CUDA
    // DENSE_QR は CUDA 未対応のため Cholesky を選択
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::CUDA;
    return;
  }
#else
  (void)useCUDA;
#endif

  // CPU 既定（従来挙動）
  options.linear_solver_type = ceres::DENSE_QR;
}

}  // namespace backend
