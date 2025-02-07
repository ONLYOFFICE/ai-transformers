# Dataset converter

> **NOTE:** requires `pandas` and `pyarrow` Python packages.

## Usage

```shell
python convert.py -o output_file.format intput_file.format
```

`.format` can be one of following: `.json`, `.jsonl` or `.parquet`.

## Samples
Convertation from JSON to JSONL:
```shell
python convert.py -o result.jsonl samples/test.json
```

Convertation from Parquet to JSON:
```shell
python convert.py -o result.json samples/test.parquet
```
etc.
