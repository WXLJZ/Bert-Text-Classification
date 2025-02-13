import logging
import sys

def get_logger(name: str) -> logging.Logger:
    r"""
    Gets a standard logger with a stream handler to stdout.
    """
    logger = logging.getLogger(name)

    # 仅当 logger 没有 handler 时，才添加 handler
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            fmt="[%(levelname)s|%(module)s:%(lineno)d] %(asctime)s - [%(name)s] >> %(message)s",
            datefmt="%m-%d-%Y %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger