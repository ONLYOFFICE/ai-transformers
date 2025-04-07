# requires following additional packages: pandas, pyarrow
import argparse
import pandas as pd
from pathlib import Path

formats = ['json', 'jsonl', 'parquet']

def get_file_name(file_path):
    return Path(file_path).stem

def data_frame_to_format(df: pd.DataFrame, output_file):
    output_format = output_file[output_file.rfind('.') + 1:]
    if output_format == 'json':
        df.to_json(output_file, orient='records')
    elif output_format == 'jsonl':
        df.to_json(output_file, orient='records', lines=True)
    elif output_format == 'parquet':
        df.to_parquet(output_file, engine='pyarrow', index=False)

def from_json(file_name, output_file):
    df = pd.read_json(file_name)
    data_frame_to_format(df, output_file)

def from_jsonl(file_name, output_file):
    df = pd.read_json(file_name, lines=True)
    data_frame_to_format(df, output_file)

def from_jsonl_to_all(input_file_path, output_dir):
    df = pd.read_json(input_file_path, lines=True)
    file_name = get_file_name(input_file_path)

    data_frame_to_format(df, f'{output_dir}/{file_name}.json')
    data_frame_to_format(df, f'{output_dir}/{file_name}.parquet')

def from_parquet(file_name, output_file):
    df = pd.read_parquet(file_name)
    data_frame_to_format(df, output_file)

handlers = {
    'json': from_json,
    'jsonl': from_jsonl,
    'parquet': from_parquet
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert datasets from one format to another.')
    parser.add_argument('--out', '-o', dest='output_file', required=True, help='specifies output file with format specified with its extension')
    parser.add_argument('input_file', help='path to the input dataset')
    args = parser.parse_args()

    input_file = str(args.input_file)
    output_file = str(args.output_file)

    input_format = input_file[input_file.rfind('.') + 1:]
    if not input_format in formats:
        raise ValueError('Wrong input format. Supported formats are: ' + formats)

    output_format = output_file[output_file.rfind('.') + 1:]
    if not output_format in formats:
        raise ValueError('Wrong output format. Supported formats are: ' + formats)

    if input_format == output_format:
        raise ValueError('The input file is already in ' + output_format + ' format')

    handlers[input_format](input_file, output_file)
