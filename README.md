# GCest
樹高推定ラスタから各種樹高成長曲線パラメータラスタを作成
Nakao et al (2022) を利用して作成される樹高推定ラスタ（任意の林齢幅における樹高を各バンドに当てはめたgeotiff）を、さまざまな樹高成長曲線（ロジスティック、ゴンペルツ、リチャーズ、対向非直角双曲線）に当てはめパラメータを推定します。非線形回帰に ceres-solver <https://github.com/ceres-solver/ceres-solver> を利用し、CPU計算に加えて GPU の利用も考慮していますが、現時点では速度向上は見込めていません。また、cmake-rcpp-template <https://github.com/jmaerte/cmake-rcpp-template> を利用してRCPP と cmake を組み合わせたプログラム作りも検討しています。
