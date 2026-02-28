import logging
from colorlog import ColoredFormatter

def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger  # уже настроен

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger