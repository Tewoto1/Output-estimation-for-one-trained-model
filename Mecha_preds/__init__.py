"""Mecha_preds -- mechanistic predictors of a trained MLP's output.

A "mechanistic predictor" takes a trained ``model.MLP`` and predicts a statistic
of its output (e.g. the output mean over Gaussian inputs) without Monte-Carlo
sampling. Predictors shipped here:

  * ``Mecha_preds.cumulants``   -- cumulant propagation (kprop): ``run_cumulants``
        (incl. an exact K=2 ReLU covariance variant), exact mean-propagation
        ``run_exact_meanprop``, and the newest shifted-weight predictor
        ``swkprop`` (``run_sw_kprop``) for the all-ones / -1/sqrt(n) weight shift.
"""
