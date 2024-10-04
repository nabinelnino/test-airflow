import logging
import sys
import os


def setup_logger(name: str = 'root', debug: bool = False, dummy: bool = False) -> logging.Logger:
    """
    Creates a logging.Logger object with the passed setup

    Parameters
    ----------
    name: str, default 'root'
        name of the logger object
    debug: bool, default False
        add DEBUG level to logger (will be INFO otherwise)
    dummy: bool, default False
        return a logging.Logger object with a NullHandler

    Notes
    -----
    Logs are always written to the CWD

    Using `dummy=True` is meant to allow for easy use of logging in function without the worry that a use might want to
     turn logging off. When this is the case, the logger will not actually log anything to a log file

    All loggers are built to catch unhandled exceptions and write them to the log file, as well as catch any
     `warnings.warn()` calls

    Returns
    -------
    logger: logging.Logger
        a Logger ready to be used for logging
    """
    logging.captureWarnings(True)  # capture all the warnings

    # return a Null logger if dummy is set to `True` since we this means we don't want to write logs
    if dummy:
        logger = logging.getLogger(name="dummy")
        logger.addHandler(logging.NullHandler())
        return logger

    logger = logging.getLogger(name=name)

    # setup handlers
    # write to package log if running in VirtualDrugBuffet directory else write to CWD
    if "PolyPharma" in os.getcwd():
        file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "polypharma.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(os.getcwd(), "polypharma.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    # set levels
    if debug:
        logger.setLevel("DEBUG")
        file_handler.setLevel("DEBUG")
        console_handler.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")
        file_handler.setLevel("INFO")
        console_handler.setLevel("INFO")

    # add handlers to logger
    logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    return logger
