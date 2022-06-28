import argparse as ag
import json

def get_parser_with_args(metadata_json_path=None):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json_path, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata

    return None
