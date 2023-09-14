from logzero import setup_logger, LogFormatter
from config.config import settings_system

logging_path = "/home/yxiao2/pycharm/GPTMan/db/log/cost.log"
formatter = LogFormatter(fmt='%(filename)s - %(asctime)s - %(levelname)s: %(message)s')
logger = setup_logger(
    logfile=logging_path,
    maxBytes=100000,
    backupCount=5,
    formatter=formatter
)

#terminal_log=setup_logger(formatter=formatter)
if __name__ == "__main__":
    #print(logging_path)
    #terminal_log.info('test')
    logger.info('test')
