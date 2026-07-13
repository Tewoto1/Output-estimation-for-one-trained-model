"""Mecha_preds -- mechanistic predictors of a trained MLP's output.

A "mechanistic predictor" takes a trained ``model.MLP`` and predicts a statistic
of its output (e.g. the output mean over Gaussian inputs) without Monte-Carlo
sampling. Predictors shipped here:

  * ``Mecha_preds.cumulants``   -- cumulant propagation (kprop): ``run_cumulants``
        (incl. an exact K=2 ReLU covariance variant), exact mean-propagation
        ``run_exact_meanprop``, the shifted-weight predictor ``swkprop``
        (``run_sw_kprop``) for the all-ones / -1/sqrt(n) weight shift, and its
        direction-general form ``spikekprop`` (``run_spike_kprop``).
  * ``Mecha_preds.binned_kprop``   -- coordinate-spike (``M = W + e1 e1^T``) binned
        K=2: HMM over the spike coordinate, one conditional bulk Gaussian per bin
        (``run_binned_kprop``; ``num_bins`` hyperparameter).
  * ``Mecha_preds.analytic_kprop`` -- ANALYTIC AFFINE K=2 for the same coordinate
        spike (writeups/analytic_affine_kprop.pdf): the bulk-given-spike law is one
        affine family ``N(mu0 + mu1 y, Sigma0 + Sigma1 y)``; the spike is quadratured
        transiently by ``num_nodes`` closed-form cells per layer
        (``run_analytic_kprop``; O(1) congruences per layer).
"""
