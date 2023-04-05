import os
import logging

log_dir = os.path.dirname(os.path.realpath(__file__))
formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d\t %(levelname)s: %(message)s')

# def setup_logger(name):
#     logging.basicConfig(level=logging.INFO,
#         filename=os.path.join(log_dir, name),
#         filemode='w', datefmt='%Y-%m-%d %H:%M:%S', format=(
#         '%(asctime)s '
#         '%(filename)s:'
#         '%(lineno)d\t'
#         '%(levelname)s: '
#         '%(message)s'
#     ))


def setup_logger(name, level=logging.INFO):
    log_file = os.path.join(log_dir, name)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger