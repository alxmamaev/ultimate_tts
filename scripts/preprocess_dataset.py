import logging
from argparse import ArgumentParser

import ultimate_tts
import yaml
from catalyst.registry import REGISTRY

logging.basicConfig(level = logging.INFO)

def parse():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--ignore_processors", default=None)

    return parser.parse_args()


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    verbose = config["dataset_preprocessing_params"]["verbose"]
    ignore_processors = (
        set(args.ignore_processors.split(","))
        if args.ignore_processors is not None
        else set()
    )

    for processor_name, processing_params in config["dataset_preprocessing_params"][
        "processors"
    ].items():
        if processor_name in ignore_processors:
            logging.info(f"Ignore {processor_name}")
            continue

        logging.info(f"Create {processor_name} processor")

        processor = REGISTRY.get_from_params(**config[processor_name])

        processor.process_files(
            processing_params["inputs"], processing_params["outputs"], verbose=verbose
        )


if __name__ == "__main__":
    args = parse()
    main(args)
