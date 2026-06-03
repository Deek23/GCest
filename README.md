# GCest
## [summary]
Nakao et al (2022) で得られる樹高推定ラスタ（任意の林齢幅における推定樹高を各バンドに当てはめた geotiffファイル）をもとに、さまざまな樹高成長曲線（ロジスティック、ゴンペルツ、リチャーズ、対向非直角双曲線）のパラメータを推定する R ライブラリです。
非線形最小二乗法計算にseres-solver <https://github.com/ceres-solver/ceres-solver> を用いて、CPU だけでなく GPU 計算も可能にしていますが、現時点では速度の向上は達成できていません。
また、 cmake-rcpp-template <https://github.com/jmaerte/cmake-rcpp-template> を利用して CPP + Cmake での R ライブラリ作りにチャレンジしています。
wsl2 上での動作は確認していますが、windows ネイティブではおそらく動作しません。

## [usage]
terra ライブラリの rast コマンドでラスタファイルを読み込み、当ライブラリをインポートしたうえで
```
  GCest(raster_object, [curveID = 1], [initparam = GCest.parameters[[curveID]]], [max_iter = 500], [useCUDA = FALSE])
```
で計算が実行されます。
ラスタオブジェクト名のみが必要な入力値で、それ以外のパラメータ（デフォルト値設定）は、
-   curveID:  樹高成長曲線（後述）
-   inputparam: パラメータの初期値
-   max_iter: 非線形最小二乗法推定の最大試行回数
-   useCUDA: seres_solver をGPUで実行するか
    -  ※ 注意： 現時点では CPUはマルチタスクで計算を行いますが、GPU計算はシングルタスクで計算するため帰って速度が低下します）
となります。

## [output]
出力はリスト形式で、
-  id: ラスタオブジェクトのセルID
-  coordinate[x, y]: ラスタオブジェクトの座標値
-  boundary[init, end]: パラメータ推定に用いられた林齢範囲
-  params[3~7 values]: 推定された樹高成長曲線パラメータ
-  convergence[1/0]: 非線二乗回帰が収束したか否か（1: 収束完了、0: 収束失敗）
-  final_cost: seles_solver の出力値（model RMSE = sqrt(2 * final_cost / N) ） * N = boundary$end - boundary$init
です。

## [height growth function]
このライブラリで用いている樹高成長曲線は、ミッチャーリッヒ式以外は
いずれも原点を通過するように修正しています。

### curveID: 1 （対向非直角双曲線、出力パラメータ：a, b, c, n, p, q）
    a * (x - c) - (m_) + sqrt((a * (x - c) - m_)^2 + 4 * a * (x - c) * m_ * q))/(2 * q) + b     (x < c)
    a * (x - c) + (n_) - sqrt((a * (x - c) + n_)^2 - 4 * a * (x - c) * n_ * p))/(2 * p) + b     (x >= c)
      ※ b = (-sqrt((m_+a*c)^2-4*a*c*m_*p)+m_+a*c)/(2*p)  
         n = b + n_

### curveID: 2 (ミッチャーリッヒ式、出力パラメータ：A, L, k)
    A * (1 - L * exp(-k * x))

### curveID: 3 (ゴンペルツ式、出力パラメータ: b, c, K)
    K * exp(- b * exp(-c * x)) - K * exp(-b)

### curveID: 4 (リチャーズ式、出力パラメータ: a, k, m)
    a * (1 - exp(-k * x)) ^ (-1/(m - 1))

### curveID: 5 (ロジスティック式、出力パラメータ a, b, m)
    m/(1 + exp(a - b * x)) - m/(1 + exp(a))

---
## License
MIT ライセンスとしています。詳細は LICENSE をご覧ください。
