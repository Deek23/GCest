// 
// ceres solver 用の 対向非直角双曲線パラメター推定
// 残差関数構造体
// ceres の example を利用して
// 
#pragma once
#include <iostream>
#include <string>
#include <cmath>
#include <armadillo>

#include <absl/log/initialize.h>
#include <ceres/ceres.h>


// 当てはめたい関数の構造体
struct ONRHResidual {
  /**
   * 引数として、x, y を配列で与える
   * x_(x) は型のキャスト（double 型 から const double型に変える）なのかね。
   * armadillo を使って、配列を参照渡しすることにしたので、引数の型を double ポインタに変更しておく。
   * 合わせてキャストへの代入も値にしておく
   */

  // コンストラクタ
  // 引数 x, y を x_, y_ に代入する
  ONRHResidual(double x, double y)
      : x_(x), y_(y) {}

  // 関数式のテンプレート
  /**対向非直角双曲線式
   * 出力パラメータとしては、a,b,c,n,p,q とするが、内部での計算は a, m_, c, n_, p, q で実施する（ceres-solverは各パラメータの個別の範囲しか指定出来ないため）
   * 
   * a*(x-c)-(m_)+sqrt((a*(x-c)-(m_))^2+4*a*(x-c)*(m_)*q))/(2*q)+b     (x < c)
   * a*(x-c)+(n_)-sqrt((a*(x-c)+(n_))^2-4*a*(x-c)*(n_)*p))/(2*p)+b     (x >= c)
   * 
   * a = p[0]                                         // 最大成長速度
   * b = (-sqrt((m_+a*c)^2-4*a*c*m_*p)+m_+a*c)/(2*p)  // 中点のy座標
   * m_ = p[1]                                        // 初期サイズ - 中点のy座標
   * c = p[2]                                         // 中点のx座標
   * n_ = p[3]                                        // 最終サイズ - 中点のy座標
   * p = p[4]                                         // 加速曲率
   * q = p[5]                                         // 減速曲率
   * n = n_ + b
   * y - f(x)  // 残差
  */
  template <typename T>
  bool operator()(const T* const p, T* residual) const {
    // p で受けたパラメータは置換できるのか？
    //double n = p[3] + p[1];
    //double m  = (p[1]*(p[1]*p[4]-p[0]*p[2]))/(p[1]-p[0]*p[2]);

    // 計算式
    // c++ はべき乗演算子が存在しないことに注意
    residual[0] = (x_ < p[2]) ?
         y_ - ((p[0] * (x_ - p[2]) - p[1] + sqrt(pow(p[0] * (x_ - p[2]) - p[1], 2.0) + 4.0 * p[0] * (x_ - p[2]) * p[1] * p[4]))/(2.0 * p[4]) + (p[0] * p[2] + p[1] - sqrt(pow(p[0] * (- p[2]) - p[1] ,2.0) + 4.0 * p[0] * (- p[2]) * p[1] * p[4]))/(2.0 * p[4])):
         y_ - ((p[0] * (x_ - p[2]) + p[3] - sqrt(pow(p[0] * (x_ - p[2]) + p[3], 2.0) - 4.0 * p[0] * (x_ - p[2]) * p[3] * p[5]))/(2.0 * p[5]) + (p[0] * p[2] + p[1] - sqrt(pow(p[0] * (- p[2]) - p[1] ,2.0) + 4.0 * p[0] * (- p[2]) * p[1] * p[4]))/(2.0 * p[4]));
    
    // 出力
    return true;
  }

 private:
  const double x_;
  const double y_;
}; 
