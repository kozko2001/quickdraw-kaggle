import argparse
from utils.config import *
from utils.seed import use_seed
from utils.cudnn import setup_cuda
from agents import *
from model import *

import logging


def main():
    logger = logging.getLogger("Main")
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    arg_parser.add_argument(
        'checkpoint',
        metavar='checkpoint',
        nargs="?",
        default=None,
        help="checkpoint to start the model loaded")
    arg_parser.add_argument('--bs', nargs='?', help='batch_size', default=-1)
    arg_parser.add_argument('-v', action='store_true')

    args = arg_parser.parse_args()
    validation_only = args.v

    # parse the config json file

    config = process_config(args.config, create_folders=not validation_only)
    if int(args.bs) > -1:
        config.batch_size = int(args.bs)

    # Create the Agent and pass all the configuration to it then run it..
    logger.info(f"config is {config}")

    if config.seed:
        use_seed(config.seed)

    if config.cuda:
        setup_cuda()

    model = globals()[config.model.cls](config)
    config.model = model


    config.checkpoint = args.checkpoint

    agent_class = globals()[config.agent]
    agent = agent_class(config)

    if not validation_only:
        agent.run()
    else:
        agent.run_validation_only()
    agent.finalize()


if __name__ == '__main__':
    main()
