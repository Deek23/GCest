/**
 * ceres-solver を利用した頑健性の高いカーブフィッティング
 * ボルツマン（ロジスティック）版
 */
#include "backend/GCest.hpp"

// 関数本体
namespace backend
{
	extern "C" OutBox estLogistic(InBox &input_data)
	{
		// boost timer を使った時間計測
		// t() の第一引数に Rcpp::Rcout とすれば、Rでも使えるかも
		boost::timer::auto_cpu_timer t(Rcpp::Rcout, 2, "%w seconds\n");

		// 推定パラメータ数
		const int n_parameter = 3;

		// 推定年数
		const int estimatedYears = input_data.Data.n_rows;

		// データ数（※ armadillo は 行をカテゴリ・グループにする方が良いので、R → Cpp の時点で転置しておくこと）
		const int kNumObservations = input_data.Data.n_cols;

		// 最大試行回数
		int maxIter = input_data.maxIteration;

		// 確認用
		Rcpp::Rcout << "estimated years: " << estimatedYears << "\n";
		Rcpp::Rcout << "sample size: " << kNumObservations << "\n";

		// CUDA 切替フラグ
		const bool useCUDA = (input_data.useCUDA != 0);

		// 結果を出力用に確保
		OutBox outbox(kNumObservations, n_parameter);

// OpenMP による並列化
// CUDA 利用時は cuBLAS/cuSOLVER ハンドルがスレッド毎に競合するため OMP を無効化
#pragma omp parallel for if(!useCUDA)
		for (int targetCol = 0; targetCol < kNumObservations; targetCol++)
		{
			//   計算対象列
			//  Rcpp:Rcout << "[col: " << targetCol + 1 << "]\n";

			// 解析用構造体作成
			NpHGdata data(input_data.Data.col(targetCol), input_data.xy.col(targetCol));

			// 初期値を変更可能な変数に代入
			// もう少しスマートな方法がありそう
			double p[n_parameter];
			for (int i = 0; i < n_parameter; i++)
			{
				p[i] = input_data.initialParameter(i);
			};

			// コスト関数
			ceres::Problem problem;

			for (int i = data.init - 20; i <= data.end - 20; ++i)
			{
				// 実際のコスト関数: AutoDiffCostFunction<最適化したい関数, 残差の次元数, パラメータ1の次元数, 2の次元数, 3の次元数...>(x_data, y_data)
				ceres::CostFunction *costFunc = new ceres::AutoDiffCostFunction<LogisticResidual, 1, n_parameter>(data.age(i), data.height(i));
				problem.AddResidualBlock(
						costFunc,										// 当てはめ用データ
						new ceres::CauchyLoss(0.5), // 残差をnullでなくてloss関数にすると、外れ値に対応出来る。
						p														// パラメータpは配列（=ポインタ）なので、変数名=アドレス
				);
				// パラメータ範囲指定
      // a
      problem.SetParameterLowerBound(p, 0, 0.01);
      problem.SetParameterUpperBound(p, 0, 1.5);
      // b
      problem.SetParameterLowerBound(p, 1, 0.05);
      problem.SetParameterUpperBound(p, 1, 0.2);
      // m
      problem.SetParameterLowerBound(p, 2, 10.0);
      problem.SetParameterUpperBound(p, 2, 80.0);
			};

			// ソルバオプション（CPU/CUDA共通ヘルパで設定）
			ceres::Solver::Options options;
			backend::configureSolver(options, maxIter, useCUDA);
			// 出力箱用意
			ceres::Solver::Summary summary;

			// 最適化実行
			ceres::Solve(options, &problem, &summary);

			// 出力用に推定パラメータをvector型に
			outbox.id(targetCol) = targetCol + 1; // R で利用するために スタートを 1にしておく
			outbox.xy.col(targetCol) = data.xy;
			outbox.estimated.col(targetCol) = arma::vec(p, n_parameter);
			outbox.boundary.col(targetCol) = {data.init, data.end};
			outbox.itteration(targetCol) = summary.termination_type == ceres::CONVERGENCE ? 1 : 0;
			outbox.result(targetCol) = summary.final_cost;

		}; // データ計算反復終了

		//出力
		return outbox;
	}
}
