import logging
from datetime import datetime
import os
LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
log_file_path = os.path.join(os.getcwd(),"LOGS",LOG_FILE)
os.makedirs(log_file_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(log_file_path, LOG_FILE)
# print(LOG_FILE_PATH)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Starting")
