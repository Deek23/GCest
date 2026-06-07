// Public host-side API for the batched CUDA solvers.
//
// The launcher converts InBox (Armadillo-backed) into flat host buffers,
// uploads them once to the device, runs a single batched LM kernel that
// solves all per-pixel problems concurrently, and writes the results back
// into OutBox.
//
// Returns false if CUDA initialization, memory allocation, or kernel launch
// fails. In that case OutBox is left untouched and the caller should fall
// back to the CPU loop.
//
// IMPORTANT: every translation unit including this header MUST have
// RcppArmadillo.h (or at minimum its config macros such as ARMA_64BIT_WORD)
// already in scope, or the arma::Mat layout inside InBox/OutBox will not
// match what other TUs see and reads will return garbage. The static_assert
// below catches the mismatch at compile time.
#pragma once

#include "InBox.hpp"
#include "OutBox.hpp"

static_assert(sizeof(arma::uword) == 8,
              "arma::uword must be 64-bit. Include RcppArmadillo.h BEFORE "
              "BatchedSolver.hpp / InBox.hpp / OutBox.hpp.");

namespace backend {

// CUDA batched LM solver for the ONRH growth curve (6 parameters).
//
// Inputs (via InBox):
//   - Data            : n_ages x n_pixels height matrix
//   - xy              : 2 x n_pixels coordinate matrix
//   - initialParameter: length-6 shared initial guess
//   - maxIteration    : per-pixel max LM iterations
//
// Outputs (via OutBox, pre-sized by caller to (n_pixels, 6)):
//   - id, xy, boundary, itteration, result, estimated
//
// Note: this routine does NOT perform the post-loop m_ -> b parameter
// transformation. The caller is responsible for that step (matches CPU path).
bool solveBatchedONRH_CUDA(const InBox& input_data, OutBox& outbox);

}  // namespace backend
