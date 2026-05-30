#pragma once
#include <string>
#include <armadillo>

// 出力用構造体
struct InBox
{
  // 変数
  arma::Mat<double> Data;
  arma::Mat<double> xy;
  arma::Col<double> initialParameter;
  int fittedCurve = 0;
  int maxIteration = 100;
  // 0: CPU (DENSE_QR), 1: CUDA (DENSE_NORMAL_CHOLESKY on GPU)
  int useCUDA = 0;


  // 基底コンストラクタ
  InBox() : Data(),
            xy(),
            initialParameter() {};

  // コンストラクタ
  InBox(int n_year, int n_col, int n_parameter) { 
      Data.set_size(n_year, n_col);
      xy.set_size(2, n_col);
      initialParameter.set_size(n_parameter);
  }
};
