// 
// ceres solver 用の リチャーズパラメター推定
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
struct RichardsResidual {
  // 引数として、x, y を配列で与える
  // x_(x) は型のキャスト（double 型 から const double型に変える）なのかね。
  // armadillo を使って、配列を参照渡しすることにしたので、引数の型を double ポインタに変更しておく。
  // 合わせてキャストへの代入も値にしておく

  // コンストラクタ
  // 引数 x, y を x_, y_ に代入する
  RichardsResidual(double x, double y)
      : x_(x), y_(y) {}

  // 関数式のテンプレート
  template <typename T>
  bool operator()(const T* const p, T* residual) const {
    // リチャーズ式：  a· (1 - exp(-k · x)) ^ (-1 / (m- 1))
    // a = p[0]
    // k = p[1]
    // m = p[2]
    // y - f(x) を残差とする
    residual[0] = y_ - p[0] * pow((1.0 - exp(-p[1] * x_)) , (-1.0 / (p[2] - 1.0)));
    return true;
  }

 private:
  const double x_;
  const double y_;
}; 
