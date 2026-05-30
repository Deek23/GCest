#pragma once
#include <omp.h>
#include <string>
#include <absl/log/initialize.h>
#include <ceres/ceres.h>
#include <boost/timer/timer.hpp>  // もう一つの時間計測

// Rcpp 関連
// RcppArmadillo も利用しているので、こちらのみをインクルードする（Rcpp.hも自動で呼ばれる）
# include "RcppArmadillo.h"

// windows用 IO
#include "export.hpp"

// 自作コンストラクタ
#include "NpHGdata.hpp"
#include "InBox.hpp"
#include "OutBox.hpp"
#include "SolverConfig.hpp"
#include "ResidONRH.hpp"
#include "ResidMitscherlich.hpp"
#include "ResidGompertz.hpp"
#include "ResidRichards.hpp"
#include "ResidLogistic.hpp"


// 計算本体

namespace backend {

	/**
	 * ceres_solver を用いた非線形最小二乗法
	 * 対向非直角双曲線版
	 * 入力用データ構造の引数。出力は Rcpp::List型
	 */
	extern "C" OutBox estONRH(InBox & input_data);

	/**
	 * ceres_solver を用いた非線形最小二乗法・樹高成長曲線推定
	 * ミッチャーリッヒ版
	 * 入力用データ構造の引数。出力は Rcpp::List型
	 */
	extern "C" OutBox estMitscherlich(InBox & input_data);

	/**
	 * ceres_solver を用いた非線形最小二乗法・樹高成長曲線推定
	 * ゴンペルツ版
	 * 入力用データ構造の引数。出力は Rcpp::List型
	 */
	extern "C" OutBox estGompertz(InBox & input_data);

	/**
	 * ceres_solver を用いた非線形最小二乗法・樹高成長曲線推定
	 * リチャーズ版
	 * 入力用データ構造の引数。出力は Rcpp::List型
	 */
	extern "C" OutBox estRichards(InBox & input_data);

	/**
	 * ceres_solver を用いた非線形最小二乗法・樹高成長曲線推定
	 * ボルツマン版
	 * 入力用データ構造の引数。出力は Rcpp::List型
	 */
	extern "C" OutBox estLogistic(InBox & input_data);
}
