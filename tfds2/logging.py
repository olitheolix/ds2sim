import logging


def getLogger(name):
    """Return a custom logger object for DS2 functionality."""
    # Return the logger object if it has handler (means we have already
    # configured it).
    logit = logging.getLogger(name)
    if logit.hasHandlers():
        return logit

    # Specify the logging format.
    fmt = (
        '%(asctime)s|%(levelname)s|%(pathname)s:%(lineno)d:%(funcName)s'
        ' (%(process)d)|%(message)s'
    )
    formatter = logging.Formatter(fmt)

    # Configure our own handler (will send the log messages to Relays) and
    # attach it to the logger object.
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logit.addHandler(handler)
    logit.setLevel(logging.DEBUG)
    return logit
