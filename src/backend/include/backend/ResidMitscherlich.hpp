// 
// ceres solver 用の ミッチャーリッヒパラメター推定
// 残差関数構造体
// ceres の example を利用して
// 
#pragma once
#include <iostream>
#include <string>
#include <armadillo>

#include <absl/log/initialize.h>
#include <ceres/ceres.h>


// 当てはめたい関数の構造体
struct MitscherlichResidual {
  // 引数として、x, y を配列で与える
  // x_(x) は型のキャスト（double 型 から const double型に変える）なのかね。
  // armadillo を使って、配列を参照渡しすることにしたので、引数の型を double ポインタに変更しておく。
  // 合わせてキャストへの代入も値にしておく

  // コンストラクタ
  // 引数 x, y を x_, y_ に代入する
  MitscherlichResidual(double x, double y)
      : x_(x), y_(y) {}

  // 関数式のテンプレート
  template <typename T>
  bool operator()(const T* const p, T* residual) const {
    // p で受けたパラメータは置換できるのか？ ← できないことが判った。
    // ミッチャーリッヒ式：  f(x) = A * (1 - L * exp(-k * x))
    // A = p[0]
    // L = p[1]
    // k = p[2]
    // y - f(x) を残差とする
    residual[0] = y_ - (p[0] * (1.0 - p[1] * exp(-p[2] * x_)));
    return true;
  }

 private:
  const double x_;
  const double y_;
}; 
