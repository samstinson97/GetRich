# AWS Lambda container for daily signal generator (3:55 PM ET weekdays)
# Build: docker build -t getrich-signal-gen .
# Test:  docker run --rm getrich-signal-gen
# Using py3.12 (Amazon Linux 2023, glibc 2.34) so manylinux_2_28+ wheels
# install without source compile. Local dev was 3.11 but all our deps
# support 3.12 cleanly.
FROM public.ecr.aws/lambda/python:3.12

# LightGBM needs libgomp.so.1 (OpenMP runtime); not bundled in the wheel.
# AL2023 uses dnf as its package manager.
RUN dnf install -y libgomp && dnf clean all

# Copy deps first (cache-friendly)
COPY production/cloud/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt -t ${LAMBDA_TASK_ROOT}/

# Copy production code
COPY production/run_all_signals.py        ${LAMBDA_TASK_ROOT}/production/run_all_signals.py
COPY production/update_dashboard.py       ${LAMBDA_TASK_ROOT}/production/update_dashboard.py
COPY production/reconcile_fills.py        ${LAMBDA_TASK_ROOT}/production/reconcile_fills.py
COPY production/audit_features.py         ${LAMBDA_TASK_ROOT}/production/audit_features.py
COPY production/dashboard.html            ${LAMBDA_TASK_ROOT}/production/dashboard.html
COPY production/order_executor.py         ${LAMBDA_TASK_ROOT}/production/order_executor.py
COPY production/risk_controls.py          ${LAMBDA_TASK_ROOT}/production/risk_controls.py
COPY production/paper_trade_tracker.py    ${LAMBDA_TASK_ROOT}/production/paper_trade_tracker.py
COPY production/config.yaml               ${LAMBDA_TASK_ROOT}/production/config.yaml
# NOTE: dashboard.py (Flask web server) is intentionally NOT included — it's for local browsing only.

# ML v3 module + model
COPY production/ml_v3/early_signal_ml_v3.py     ${LAMBDA_TASK_ROOT}/production/ml_v3/early_signal_ml_v3.py
COPY production/ml_v3/shap_monitor.py           ${LAMBDA_TASK_ROOT}/production/ml_v3/shap_monitor.py
COPY production/ml_v3/config.yaml               ${LAMBDA_TASK_ROOT}/production/ml_v3/config.yaml
COPY production/ml_v3/refresh_cache.py          ${LAMBDA_TASK_ROOT}/production/ml_v3/refresh_cache.py
COPY production/ml_v3/universe_live.csv         ${LAMBDA_TASK_ROOT}/production/ml_v3/universe_live.csv

# ML v4 module + model
COPY production/ml_v4/early_signal_ml_v4.py     ${LAMBDA_TASK_ROOT}/production/ml_v4/early_signal_ml_v4.py
COPY production/ml_v4/shap_monitor.py           ${LAMBDA_TASK_ROOT}/production/ml_v4/shap_monitor.py
COPY production/ml_v4/config.yaml               ${LAMBDA_TASK_ROOT}/production/ml_v4/config.yaml
COPY production/ml_v4/audit_features.py         ${LAMBDA_TASK_ROOT}/production/ml_v4/audit_features.py
COPY production/ml_v4/place_pending_stops.py    ${LAMBDA_TASK_ROOT}/production/ml_v4/place_pending_stops.py
COPY production/ml_v4/universe_live.csv         ${LAMBDA_TASK_ROOT}/production/ml_v4/universe_live.csv

# Patch Windows hardcoded paths in config.yaml to Lambda Linux paths
RUN sed -i 's|C:\\\\ai-research-team\\\\v10_ml_v3\\\\seed_6|/var/task/v10_ml_v3/seed_6|g' \
    ${LAMBDA_TASK_ROOT}/production/ml_v3/config.yaml && \
    sed -i 's|C:\\\\ai-research-team\\\\v10_ml_v4_sortino\\\\saved_model_1188_s6|/var/task/v10_ml_v4_sortino/saved_model_1188_s6|g' \
    ${LAMBDA_TASK_ROOT}/production/ml_v4/config.yaml

# v3 model files (134 MB)
COPY v10_ml_v3/seed_6/                    ${LAMBDA_TASK_ROOT}/v10_ml_v3/seed_6/

# v4 model files (88 MB)
COPY v10_ml_v4_sortino/saved_model_1188_s6/  ${LAMBDA_TASK_ROOT}/v10_ml_v4_sortino/saved_model_1188_s6/

# v10_2_engine for FACTOR_NAMES (small)
COPY v10_2_engine.py                      ${LAMBDA_TASK_ROOT}/v10_2_engine.py

# Cloud-specific Lambda handler
COPY production/cloud/lambda_handler.py   ${LAMBDA_TASK_ROOT}/lambda_handler.py
COPY production/cloud/s3_state.py         ${LAMBDA_TASK_ROOT}/s3_state.py

# Lambda needs ${LAMBDA_TASK_ROOT} on PYTHONPATH (default) and CMD = "module.handler"
CMD ["lambda_handler.handler"]
