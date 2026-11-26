# Slide: Production Scalability, Drift Monitoring & Continuous Retraining

## Headline: From Prototype to Enterprise: Sustainable ML Lifecycle

### 1. Architectural Evolution Path
- **Database**: In-memory SQLite prototype ➔ Managed PostgreSQL / Snowflake enterprise warehouse with partition pruning and customer change data capture (CDC).
- **Serving Layer**: Streamlit interactive app ➔ Asynchronous FastAPI microservice containerized on Kubernetes, deployed behind CRM / frontline agent desktop with sub-50ms inference.
- **Scoring Pipeline**: Scheduled daily/weekly batch scoring via Celery / Apache Airflow into `churn_risk_scores` table; event-driven webhook triggering immediate real-time rescore on contract renewal or cancellation inquiry.

### 2. Automated Drift Monitoring (PSI Engine)
- **Population Stability Index (PSI)** computed on weekly ingestion batches across numeric & categorical features:
  - `PSI < 0.10`: Stable distribution (No intervention required).
  - `0.10 ≤ PSI < 0.25`: Moderate feature distribution shift (Slack / PagerDuty Warning).
  - `PSI ≥ 0.25`: Significant population drift (Trigger automated retrain & threshold recalibration alert).
- **Concept & Calibration Gap Tracking**: Continuous monitoring of actual vs. predicted churn rate to detect macroeconomic customer behavioral shifts.

### 3. Quarterly Retraining & Profit Re-Optimization
- **Simulated Drifted Batch**: Shifted distribution (inflationary bill shocks, contract lapses, tech support drop) moves churn from 26.5% to ~37.1%.
- **Automated Threshold Recalibration**: Deployed model adapts operating cutoff (0.35 ➔ 0.20) dynamically maximizing net profit curve.
- **ROI Impact**: Dynamic threshold re-optimization protects retention margins and captures incremental profit without requiring emergency manual re-architecture.
