# Artifact and claim map

| Question | Main method | Primary evidence | Validation |
|---|---|---|---|
| Q1 | HGBR + seasonal baselines; deterministic non-preemptive scheduling | `outputs/tables/q1_forecast_metrics.csv`, `q1_schedule_final24.csv` | `q1_constraint_validation.json` |
| Q2 | Cost–carbon task-placement trade-off, storage disabled | `q2_policy_comparison.csv`, `q2_schedule_final24.csv` | `q2_constraint_validation.json` |
| Q3 | Six independent sparse storage/electricity LPs | `q3_storage_effects.csv`, `q3_solver_status.json` | solver residuals, SOC and simultaneity checks |
| Q4 | Fixed-mode task/energy coordination; carbon-price and renewable scenarios | `q4_scenario_results.csv`, recommendation task/energy tables | `outputs/run_manifest.json` |

The full pipeline, data anomalies, claim boundaries, and all figure paths are described in `README.md`, `docs/model.md`, `docs/reproducibility.md`, and `paper/solution_report.pdf`.

Raw contest workbooks are intentionally excluded from the public repository; place authorized copies under `data/` to rerun the pipeline.
Data-derived reports, schedules, figures, and tables are generated locally and intentionally excluded from the public repository unless their redistribution is separately authorized.
