import logging, sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_format = "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(file)s"
formatter = jsonlogger.JsonFormatter(log_format)

console = logging.StreamHandler(sys.stderr)
console.setFormatter(formatter)

file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "app.log",
    when="midnight",
    backupCount=14,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console, file_handler]
)

logger = logging.getLogger(__name__)

