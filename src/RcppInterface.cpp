#include "backend/GCest.hpp"

using namespace Rcpp;

/**
 *
 *  おそらく、[[Rcpp::export]] を利用しないで Rにインポートするための手順だと思われる。
 *  R -> C++ への引数渡し、結果出力の手順は基本ここで行う。
 *
 *  Rcpp functions are functions that do have defined input/output Rcpp-Types.
 *
 * NOTE: Normally Rcpp functions are auto-wrapped by Rcpp::compileAttributes() in an RcppExports.cpp,
 *  but since we want to maximize platform-independence and some systems get an
 *  enterRNGScope error in RcppExports.o when using [[Rcpp::export]] we wrap the functions manually.
 *
 */

/**
 *  backend の関数（計算本体）への入出力を実施
 *  前段で SEXP からの変換を済ましているので、ここでは単にバックエンド関数に渡すだけ（もはや必要無いかも）
 */
extern "C" OutBox IO_GCest(InBox &data_)
{
  // 当てはめ関数ID
  //[1: 対向非直角双曲線, 2: ミッチャーリッヒ...]
  const std::string growthCurve[6] = {"null", "ONRH", "Mitscherlich", "Gompertz", "Richards", "Logistic"};

  // 樹高成長曲線
  const int curveID = data_.fittedCurve;

  Rcout << "step 2...fitted curve: " << growthCurve[curveID] << "\n"; // 確認用

  // 出力用変数
  OutBox out;

  // 樹高成長曲線毎に当てはめる関数
  switch (curveID)
  {
  case 0:
  {
  Rcpp::Rcout << "invarid curveID\n";
    break;
  }
  case 1:
  {
    out = backend::estONRH(data_);
    break;
  }
  case 2:
  {
    out = backend::estMitscherlich(data_);
    break;
  }
  case 3:{
    out = backend::estGompertz(data_);
    break;
  }
  case 4:{
    out = backend::estRichards(data_);
    break;
  }
  case 5:{
    out = backend::estLogistic(data_);
    break;
  }
  default:
  {
    Rcpp::Rcout << "invarid curveID\n";
    break;
  }
  }
  // 出力
  return out;
}

/**
 *  R からの引数受け取りと Rへの出力受け渡しを実施
 *  R -> Rcpp の入出力型は SEXP型なので、これでやりとりするためのラッパ
 *  R -> Rcpp では、 data.frame -> DataFrame としておく
 *  SEXP WRAPPER SECTION.
 *  SEXP wrappers are functions whose sole purpose is redirection of inputs and outputs to and from Rcpp functions.
 *
 */
extern "C" SEXP GCest(SEXP _data, SEXP _xy, SEXP _parm, SEXP gcID, SEXP maxIter, SEXP useCUDA)
{
  // データ変換
  arma::mat Data = Rcpp::as<arma::mat>(_data);
  arma::mat xy = Rcpp::as<arma::mat>(_xy);
  arma::vec initialParameter = Rcpp::as<arma::vec>(_parm);

  int n_year = Data.n_rows;
  int n_col = Data.n_cols;
  int n_parameter = initialParameter.n_elem;
  // ここで R list ↔ Rcpp::list の相互変換
  // Rcpp::as を使って任意の形に変換
  InBox data(n_year, n_col, n_parameter);
  data.Data = Rcpp::as<arma::mat>(_data);
  data.xy = Rcpp::as<arma::mat>(_xy);
  data.initialParameter = Rcpp::as<arma::vec>(_parm);
  data.fittedCurve = Rcpp::as<int>(gcID);
  data.maxIteration = Rcpp::as<const int>(maxIter);
  data.useCUDA = Rcpp::as<int>(useCUDA);

  Rcout << "step 1 O.K.\n"; // 確認用
  // そのまま実行コードインターフェイスへ
  OutBox outbox = IO_GCest(data);

  // 出力
  Rcpp::List out = Rcpp::List::create(
      _["id"] = outbox.id,
      _["coordinate"] = outbox.xy,
      _["boundary"] = outbox.boundary,
      _["params"] = outbox.estimated,
      _["convergence"] = outbox.itteration,
      _["final_cost"] = outbox.result); // 変数名を着ける場合は _["変数名"]

  return Rcpp::wrap(out);
}

/**
 *
 *  INIT SECTION.
 *
 *  When wrapping your functions via SEXP manually you have to explicitly tell Rcpp how
 *  many SEXP arguments your wrapper functions have and register them under an export name.
 *  主導でSEXP設定する場合に実際に幾つの変数を使うかを宣言しておく
 */
static const R_CallMethodDef CallEntries[] = {
    {"GCest", (DL_FUNC)&GCest, 6},
    {NULL, NULL, 0}};

extern "C" void R_init_GCest(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
