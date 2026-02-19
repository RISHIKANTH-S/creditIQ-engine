import logging
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    filename=log_filepath,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger("CreditIQ")
