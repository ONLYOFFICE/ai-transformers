# requires following additional packages: pandas, pyarrow
import argparse
import sys
import os
sys.path.append('../../../build_tools/scripts/sdkjs_common/jsdoc')
import generate_jsonl_dataset
import convert

def generate(output_dir, model):
    output_dir = os.path.abspath(output_dir)
    os.chdir('../../../build_tools/scripts/sdkjs_common/jsdoc')
    generate_jsonl_dataset.generate(output_dir, model)
    file_name = 'dataset.jsonl'
    convert.from_jsonl_to_all(f'{output_dir}/{file_name}', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate documentation JSONL dataset")
    parser.add_argument(
        "destination", 
        type=str, 
        help="Destination directory for the generated documentation",
        nargs='?',  # Indicates the argument is optional
        default="office-js-api"  # Default value
    )
    parser.add_argument(
        "model", 
        type=str, 
        help="Type of model",
        nargs='?',  # Indicates the argument is optional
        default=""  # Default value
    )
    args = parser.parse_args()

    generate(args.destination, args.model)
