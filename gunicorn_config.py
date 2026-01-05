import os

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Gunicorn configuration
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 300  # 5 minutes - allows trades to complete
graceful_timeout = 120  # 2 minutes for graceful shutdown
keepalive = 5

# Logging
accesslog = f"{LOG_DIR}/gunicorn_access.log"
errorlog = f"{LOG_DIR}/gunicorn_error.log"
loglevel = "info"

# Worker lifecycle
max_requests = 1000
max_requests_jitter = 50