GCest.parameters <- list(
    "ONRH"         = c(1.0, 6.0, 10.0, 30.0, 0.50, 0.90)  # ONRH (a, m_, c, n_, p, q)
  , "Mitscherlich" = c(40.0, 1.0, 0.07)  # Mitscherlich (a, b, k)
  , "Gompertz"     = c(40.0, 2.0, 0.1)  # Gompertz (K, b, c)
  , "Richards"     = c(40.0, 0.7, 0.07)  # Richards (a, k, m)
  , "Logistic"    = c(1.0, 0.07, 40.0)  # Logistic (a, b, m)
)

GCest <- function(inputRaster, curveID = 1, initparam = GCest.parameters[[curveID]], max_iter = 500, useCUDA = FALSE) {
  if (class(inputRaster) != "SpatRaster") stop(gettextf("data should be SpatRaster"));
  # 推定林齢数
  n.year <- terra::nlyr(inputRaster);
  # 樹高・座標データを作る
  tmp <- t(cbind(terra::as.matrix(inputRaster), terra::crds(inputRaster, na.rm = FALSE)));
  tmp <- tmp[, ! is.na(colSums(tmp))];
  # 計算
  .Call("GCest", tmp[1:n.year, ], tmp[(n.year + 1):(n.year + 2), ], initparam, curveID, max_iter, as.integer(useCUDA));
}
