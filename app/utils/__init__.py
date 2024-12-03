import logging
def print_debug(s, DEBUG=False):
    if DEBUG:
        logging.info(print(s))
        print(s)
