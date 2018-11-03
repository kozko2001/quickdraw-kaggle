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
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    logger.info(f"config is {config}")

    if config.seed:
        use_seed(config.seed)

    if config.cuda:
        setup_cuda()

    model = globals()[config.model.cls](config)
    config.model = model

    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
