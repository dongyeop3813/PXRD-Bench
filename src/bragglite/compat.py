"""Historical wire identifiers and RNG domains for existing checkpoints.

These values are deliberately immutable: changing a seed domain changes samples,
and changing a schema identifier prevents existing artifact readers from resuming.
All public model, config, entrypoint and environment names use Bragglite.
"""

EVAL_SEED_DOMAIN = 'crystalite-eval'
CSP_SEED_DOMAIN = 'crystalite-csp-candidate-v2'
SHARED_ENCODERS_SCHEMA = 'crystalite_shared_phase_encoders_v1'
RUN_SCHEMA = 'crystalite_run_v2'
METRICS_SCHEMA = 'crystalite_metrics_v1'
EVALUATION_SCHEMA = 'crystalite_evaluation_run_v2'
PREDICTION_CACHE_SCHEMA = 'crystalite_prediction_cache_npz_v1'
