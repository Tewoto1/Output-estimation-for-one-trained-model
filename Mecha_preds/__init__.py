"""Mecha_preds -- mechanistic predictors of a trained MLP's output.

A "mechanistic predictor" takes a trained ``model.MLP`` and predicts a statistic
of its output (e.g. the output mean over Gaussian inputs) without Monte-Carlo
sampling. Predictors shipped here:

  * ``Mecha_preds.cumulants``   -- cumulant propagation (kprop), incl. an exact
        K=2 ReLU covariance variant, the spike-aware ``skprop`` and symbolic
        ``shkprop`` sub-predictors.
  * ``Mecha_preds.clippedProp`` -- structured propagation that carries the
        all-ones (mean-shift) channel as an explicit clamped-Gaussian scalar plus
        a perpendicular Gaussian; built for shifted-/clamped-mean inputs.
"""
