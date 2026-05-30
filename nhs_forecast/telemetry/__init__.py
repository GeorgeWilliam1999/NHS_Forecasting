"""Pay-per-use telemetry & utilisation underwriting layer.

This subpackage is self-contained and additive: it does not change the existing
aggregate procedure→equipment demand pipeline. It supplies the device-session
grain, censored count modelling, and portfolio risk/underwriting that the
aggregate pipeline cannot provide.

Modules
-------
synthetic   : deterministic device-session telemetry event-log generator
sessionise  : reconstruct sessions + device-day aggregates from the event log
features    : leakage-safe device-day feature engineering
model        : censored Negative-Binomial baseline + pinball-loss backtest
risk        : factor covariance, Monte-Carlo portfolio risk, pricing
pipeline     : orchestration + artifact persistence (read by the dashboard)
"""
