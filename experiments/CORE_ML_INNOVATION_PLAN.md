# Core ML Innovation & Kaggle Grandmaster Plan
## Event-Driven Congestion Impact Forecasting (Planned + Unplanned)

**Date**: 2026-06-23  
**Focus**: Pure core ML/FE/optimization/experimentation. **Frontend, dashboard, Streamlit completely out of scope.**  
**Guiding Rule**: Work iteratively. Never break or overwrite existing production paths (especially GridGuardV9Inference, V9 models, precompute.py, data_pipeline, legacy inference).

---

## 1. Current State Assessment (Deep Analysis)

### 1.1 Production-Quality Duration Engine (Very Strong)
- `src/inference.py` → `GridGuardV9Inference` (aliased as GridGuardInference for compatibility)
  - 3-model ensemble: LGB (L1), LGB DART, XGBoost on **log1p(duration_hrs)** target.
  - Extremely rich, carefully engineered features (V9):
    - Multiple high-signal LOO encodings (corridor_loo, station_loo, zone_cause_loo, veh_cause_loo, **cluster_dur_loo** (top in SHAP), officer_dur_loo, pin_code_loo).
    - DBSCAN cluster features at inference time (vectorized nearest cluster).
    - Address-derived: is_junction, addr_road_class, pin extraction.
    - Officer workload signals.
    - Text (TFIDF + SVD, description_len).
    - Mappls optional enrichment.
    - Monotone constraints in training (V9 experiment).
  - OOF inverse-MAE weighted blending.
  - Training driver: `experiments/gridguard_v9.py` — textbook Kaggle GM practices:
    - Vectorized (no iterrows) LOO.
    - Strict chronological sorting.
    - Careful duration computation (closed > resolved > censored).
    - Optuna (150 trials), feature pruning based on SHAP=0.
    - Increased SVD dims, etc.

**V9 represents serious engineering. We must treat it as the gold-standard baseline for duration and reuse its signals.**

### 1.2 Current Impact / "Event-Driven" Work (Fragmented, Nascent)
- `src/advanced_event_fe.py` + `src/event_impact_predictor.py`
  - Separate FE focused on:
    - Planned density per corridor (`corridor_planned_density`).
    - Regime flags (`is_planned`, `is_planned_cause`).
    - Keyword text flags + SVD.
    - Spatial distances to major junctions.
    - High-order planned × rush/centrality interactions.
    - Composite `impact_target` construction.
  - Trains its own models: `lgb_event_dur_quantile`, `lgb_event_closure`, `lgb_event_impact`.
  - Rule-based recommendation layer (officers, barricade, diversion, cascade_watch).
- New artifact: `data/precomputed/corridor_impact_stats.parquet`.
- Training script: `experiments/event_impact_grandmaster_v1.py` (temporal CV, log targets, but metrics suffered from leakage + not leveraging V9 features).

**Problems**:
- Duplication of text/spatial/encoding logic.
- Does **not** consume V9's superior LOO, cluster, officer features.
- Standalone models → fragmentation.
- Recommendations are heuristic (good start, not GM).
- Precompute.py does **not** include impact artifacts.
- Planned events (~5.7%) are under-modeled as a regime.

### 1.3 Other Supporting Components (Good Primitives)
- Cascade detection, SpatioTemporalForecaster (DBSCAN + risk tensor), TriageOptimizer (MILP), closure predictor, survival (KM).
- Data pipeline has solid censoring fixes.
- Lots of historical experiments (v1–v9) with SHAP, Optuna, quantile work — good institutional knowledge.

### 1.4 Data Realities (Critical for GM Design)
- Planned events rare + different distribution (higher closure declaration, off-peak skew, specific causes: construction/public_event/procession).
- Heavy right tail + censoring.
- Strong autocorrelation → must use proper temporal/purged CV.
- "Impact" is not directly observed; must be constructed (duration × capacity reduction × network load).

---

## 2. Core Problem Reframing (Innovation North Star)

**Not** "predict how long an event will last."

**Instead**:
> Given a planned or unplanned traffic event (with partial real-time info), forecast its **congestion impact footprint** on the network and produce **defensible, optimized resource recommendations** (officers to deploy, barricading level, diversion/watch strategy) that BTP can act on before or during the event.

Key decision variables:
- Manpower allocation under constraint.
- Barricade yes/no + intensity.
- Diversion focus areas.
- Secondary (cascade) monitoring.

This requires:
- Better primary target(s) than raw duration.
- Regime awareness (planned events can be acted on *in advance*).
- Uncertainty-aware outputs.
- Link from prediction → actionable optimization.

---

## 3. Kaggle Grandmaster Principles We Will Apply

1. **Causal & Temporal Integrity** — Every feature and CV split must be impossible to leak future info. Use purged/grouped time CV where needed.
2. **Extract Signal from Rarity** — Planned events are the highest-ROI subset. Techniques: importance weighting, synthetic data, representation sharing, two-stage or mixture models.
3. **Target Engineering is 50% of the win** — Define "Congestion Impact Units (CIU)" rigorously. Consider multi-objective or learned impact.
4. **Reuse + Compose** — V9 is excellent. New work should **consume** V9 predictions/features rather than duplicate.
5. **Systematic Experimentation** — Ablations, regime splits (planned vs unplanned), error analysis by cause/corridor, SHAP per model.
6. **Decision-Centric Modeling** — Move from "predict X then rule" to models/optimization whose loss approximates downstream cost (officer deployment cost vs prevented delay).
7. **Uncertainty & Safety** — Quantile + conformal prediction for manpower ranges ("safe to deploy between 4-7").
8. **Reproducibility & Versioning** — Every experiment run produces traceable artifacts (features list, params, CV scores, model files with version).
9. **Leverage Existing Brilliance** — DBSCAN clusters, officer LOO, cascade logic, Mappls, triage MILP are assets.

---

## 4. Detailed Iterative Phased Plan (Safe, Additive Only)

### Phase 0 — Audit & Documentation (Current)
- [x] Full re-analysis of V9 + impact additions + precompute + data.
- [ ] Write this PLAN + summary of gaps.
- Output: `experiments/CORE_ML_INNOVATION_PLAN.md` (this file) + perhaps a short `experiments/current_state_audit.md`.

**Do not change any code in this phase.**

### Phase 1 — Non-Breaking Shared Foundation (Safe Reusability)
**Goal**: Stop duplication. Make V9's best ideas available to impact work.

**Actions (all new files or careful edits)**:
1. Create `src/core_features.py` (brand new file):
   - Vectorized LOO table builder (inspired by V9 but generalized).
   - Cluster predictor class (reuse dbscan_v9).
   - Officer + pin + zone_cause LOO builders.
   - Safe temporal feature creator.
   - Function to enrich a dataframe with "V9-style" features given raw data + fitted tables.
2. Refactor `advanced_event_fe.py` (in-place, additive):
   - Add optional `use_v9_features=True` path that calls into core_features.
   - Keep old behavior when flag is off.
   - Add planned_density as first-class (already good).
3. Create `experiments/shared_experiment_utils.py`:
   - `make_temporal_splits(df, n_splits=5, purge_days=2)` (purged CV).
   - `evaluate_regime(df, preds, y, regime_col='is_planned')`.
   - Result logger + artifact saver with versioning.
4. Update `experiments/event_impact_grandmaster_v1.py` minimally or leave it; new v2 will use the harness.

**Rules**:
- No changes to `src/inference.py`, `gridguard_v*.py`, or any existing model files.
- New core_features must be importable without side effects.

**Deliverable**: Phase 1 PR-like branch of new files + tests that old V9 inference still works unchanged (smoke test).

### Phase 2 — Superior Targets + Planned Regime Focus
**Define 1st-class targets**:
- `log_duration` (keep for compatibility).
- `closure` (already strong signal).
- `congestion_impact` = duration_hrs × (1 + 1.5*requires_road_closure) × corridor_centrality × (1 + 0.6*is_rush) × volume_proxy.
- Optional: `secondary_risk` (cascade probability as auxiliary).
- Perhaps learned "disruption" if we can construct better proxies from concurrent events.

**Planned-specific innovations**:
- Higher sample weight for planned rows.
- Synthetic planned augmentation (perturb known planned events with realistic variations in hour/scale).
- Regime embedding or separate "planned head".
- Features that only make sense for planned (advance notice proxy, expected scale from cause + corridor history).

**Experiment**:
- `experiments/impact_v2_target_experiment.py`
- Report: overall impact MAE + planned-subset metrics vs naive (duration * heuristics) and vs v1.

### Phase 3 — Advanced GM Modeling & Optimization Experiments
Ideas to iterate (one vN at a time):
- Multi-task LightGBM (or two LGBs sharing leaves) predicting (duration, impact, closure).
- Stacking: V9 duration prediction (from GridGuardV9Inference, run in inference mode on train folds) + new planned features → impact head.
- Optuna on impact objective + monotonic constraints on key LOO/centrality features.
- Feature selection (permutation importance or Boruta-like on top of V9 base).
- Quantile + conformal prediction wrapper for actionable intervals.
- Cascade as graph or intensity feature (enhance existing cascade_detector).

**Evaluation Harness** (mandatory):
- Temporal purged CV.
- Metrics: MAE (impact), AUC (closure), regime-specific scores, "regret" simulation (if we had deployment logs).
- Ablation table for every experiment.

**Files**: `experiments/impact_v3_multitask.py`, `experiments/impact_v4_stacked.py`, etc.
Save distinct artifacts: `lgb_impact_v3_*.pkl`, `impact_v3_features.json`.

### Phase 4 — Learned / Optimized Recommendation Layer
Move beyond heuristics:
- Option A (light): Train a small model or use gradient boosting to predict "optimal officers given impact, closure, zone_capacity".
- Option B (GM): Formulate as small optimization problem (use scipy.optimize or PuLP if added, or simple greedy + local search) that takes **multiple** predicted events + officer budget per zone and solves allocation to minimize total predicted impact.
- Integrate existing `triage_optimizer.py` (MILP) and extend it with impact scores instead of just STIS.
- Output not just point recommendations but ranges ("4-6 officers gives 80% coverage of expected impact").

**Simulation Eval**: Create hold-out "what-if" scenarios (multiple concurrent events) and score different policies.

### Phase 5 — Full Experimentation Loop + Diagnostics
- Central runner that can execute a config of experiments.
- Automatic SHAP on best model per regime.
- Hard-case analysis: long-duration construction events, high-centrality planned events.
- Versioned results in `experiments/results/impact_*/`
- Model card stub for the best impact model.

### Phase 6 — Additive Integration (Optional, Later)
- New file `src/impact_forecaster.py`:
  ```python
  from src.inference import get_inference_engine  # V9
  class ImpactForecaster:
      def predict_full(self, event):
          v9 = v9_engine.predict(event)
          extra = self.fe.transform(...)
          impact = self.impact_head.predict( {**v9, **extra} )
          recs = self.recommend(v9, impact, ...)
          return {**v9, "impact": impact, "recommendations": recs}
  ```
- This can be used by anything that wants impact/recs without touching the duration engine.
- Update precompute only if we want to materialize new tables (additive step).

---

## 5. Strict Non-Breaking Rules

- **Never** edit: `src/inference.py`, any `gridguard_v*.py`, `src/precompute.py` (unless purely additive import at the end), existing v9_*.pkl or v9_features.
- New training scripts live in `experiments/`.
- New model files get distinct names with version (e.g. `lgb_impact_v2_final.pkl`).
- When enhancing `advanced_event_fe.py`, preserve exact previous behavior when new flags are off.
- Every change must have a smoke test that the old `GridGuardV9Inference().predict(...)` still returns the same shape/keys.
- Use `experiments/results/` with dated or versioned subdirs.
- Log random seeds, data hash, code version for every run.

---

## 6. Success Metrics (for the core innovation)

1. **Impact prediction quality**: MAE on composite impact (or rank correlation) significantly better than "V9_duration * static multiplier".
2. **Planned regime lift**: Relative improvement on the planned subset >= overall improvement (or better).
3. **Recommendation quality**: In a simple multi-event simulation, allocation using predicted impact reduces total "regret" vs using only duration or STIS.
4. **Reusability**: V9 features + new planned/impact features can be composed cleanly (measured by reduced code duplication).
5. **Experiment hygiene**: Every major idea has an ablation showing its contribution.

---

## 7. Execution Order (Iterative)

1. **Write this plan + quick audit summary** (done).
2. **Phase 1** — Implement safe `core_features.py` + update to advanced_event_fe + harness skeleton. Verify nothing broke.
3. Run baseline "v1 impact" vs new shared features to prove value.
4. Phase 2 target definition + first strong v2/v3 experiment.
5. Continue phases with one focused experiment + diagnostic at a time.
6. Only after strong internal validation, consider the additive integration in Phase 6.

**Use todo tracking for each phase and subtask.**

---

## 8. Open Questions / Risks (to resolve iteratively)

- How best to "inject" V9 predictions as features without circularity in CV (use out-of-fold V9 predictions during training of impact head).
- Best construction of the impact target (involve domain constants from ISEC study + corridor stats we already have).
- Whether to treat "planned" as a known flag at inference time for all cases (yes for the "planned events" page; maybe inferred for others).
- Adding true graph features (osmnx) — high value but expensive; do in a later sub-experiment.

---

**Next Immediate Step After Plan Approval**: Start Phase 1 implementation using only new files + minimal safe edits.

This plan respects the existing high-quality V9 work while pushing hard on the actual user problem (event-driven congestion decisions) with true grandmaster rigor.