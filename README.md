# GCest
Nakao et al (2022) で得られる樹高推定ラスタ（任意の林齢幅における推定樹高を各バンドに当てはめた geotiffファイル）をもとに、さまざまな樹高成長曲線（ロジスティック、ゴンペルツ、リチャーズ、対向非直角双曲線）のパラメータを推定する R ライブラリです。
非線形最小二乗法計算にseres-solver <https://github.com/ceres-solver/ceres-solver> を用いて、CPU だけでなく GPU 計算も可能にしていますが、現時点では速度の向上は達成できていません。
また、 cmake-rcpp-template <https://github.com/jmaerte/cmake-rcpp-template> を利用して CPP + Cmake での R ライブラリ作りにチャレンジしています。
wsl2 上での動作は確認していますが、windows ネイティブではおそらく動作しません。

---
## License
MIT ライセンスとしています。詳細は LICENSE をご覧ください。
