//
// npHG tiff の出力値から成長曲線解析向けの構造体を作る
// fvec XY: セルの座標
// fvec age: 20~100
// fvec height: 20~100年生のモデル推定樹高
// bool vec isUse: 解析に使う/使わない の判断値（20年生時の樹高と同一の樹高, 100年生時の樹高と同一の樹高データは除外するようにする）
//
#pragma once
#include <armadillo>
#include <string>

struct NpHGdata
{
  // 変数
  arma::Col<double> age = arma::linspace(20.0,100.0, 81);
  arma::Col<double> height;
  arma::Col<double> xy;
  int init = 20;
  int end = 100;
  int years = 80;

  // 基底コンストラクタ
  NpHGdata() : height(81, arma::fill::zeros),
               xy(2, arma::fill::zeros) {};

  // 列ベクタで与えた場合コンストラクタ
  NpHGdata(arma::vec height_, arma::vec xy_){
    // 推定林齢数
    int years = height_.n_elem -1;
    // 樹高データ
    height = height_;
    // 座標データ
    xy = xy_;
    // 林齢
    age = age.subvec(0,years);

    // 有効データ範囲計算用
     int i = 0;
     int j = years - 1;

    // 解析に使うデータの開始位置
    while (height(i) == height(0))
    {
      i++;
    };
    init = i + 19; // 林齢に換算して保持
    // 解析に使うデータの終了位置
    while (height(j) == height(years - 1))
    {
      j--;
    };
    end = j + 21; // 林齢に換算して保持
  };
};
