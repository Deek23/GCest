#pragma once
#include <string>
#include <armadillo>

// 入力用構造体
struct OutBox
{
  // 変数
  arma::Col<int> id;
  arma::Mat<double> xy;
  arma::Col<double> result;
  arma::Col<int> itteration;
  arma::Mat<int> boundary;
  arma::Mat<double> estimated;

  // 基底コンストラクタ
  OutBox() : id(5),
             xy(2, 5),
             result(5),
             itteration(5),
             boundary(2, 5),
             estimated(6, 5) {};

  // コンストラクタ
  OutBox(int n_col, int n_parameter) : id(n_col, arma::fill::zeros),
                                       xy(2, n_col, arma::fill::zeros),
                                       result(n_col, arma::fill::zeros),
                                       itteration(n_col, arma::fill::zeros),
                                       boundary(2, n_col, arma::fill::zeros),
                                       estimated(n_parameter, n_col, arma::fill::zeros) {};

  // JSON への変換用マクロ
  // nlohmann::NLOHMANN_DEFINE_TYPE_INTRUSIVE(OutBox, id, result, itteration, boundary, estimated);

  // csv形式で出力
  void save (std::string filename){
    arma::Mat<double> out(id.n_rows, 7+estimated.n_rows);
    out.col(0) = arma::conv_to <arma::colvec>::from(id);
    out.cols(1,2) = arma::conv_to <arma::colvec>::from(xy);
    out.cols(3,4) = arma::conv_to <arma::colvec>::from(boundary);
    out.col(5) = arma::conv_to <arma::colvec>::from(itteration);
    out.col(6) = result;
    out.cols(7, 7+estimated.n_rows) = estimated;

    // 出力
  };
};
