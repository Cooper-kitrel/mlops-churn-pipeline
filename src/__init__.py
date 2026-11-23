import os

# Silence an MLflow 3 informational log line
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
