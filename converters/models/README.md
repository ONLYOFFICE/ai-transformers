# Model converter

Since model convertation is a sophisticated and complex process, the convertation from `.safetensors` to `.gguf` is done with [llama.cpp](https://github.com/ggerganov/llama.cpp) scripts.

To download the last version of this script, call:

```shell
python get_converter.py
```

This will download `convert_hf_to_gguf.py` and auxiliary `convert_hf_to_gguf_update.py`.

> **NOTE:** before using the script install all requirements with:

```shell
pip install -r requirements.txt
```

## Usage

To convert `.safetensors` model to `.gguf`, use `convert_hf_to_gguf.py` script as following:

```shell
python convert_hf_to_gguf.py model --outfile OUTFILE
```
 - `model` - the directory containing model file
 - `OUTFILE` - output model name with `.gguf` extension

The script provides variety of settings including quantization via `--outtype` flag. To see more options run:

```shell
python convert_hf_to_gguf.py -h
```
or head to official documentation in [llama.cpp](https://github.com/ggerganov/llama.cpp) repo.

## convert_hf_to_gguf_update.py

Models with BPE tokenization use pre-tokenizers which are often different for various BPE models. There is currently no standardized and consistent way to provide information about the pre-tokenizer used by a model, but yet this information should be in the "tokenizer.ggml.pre" entry of the resulted GGUF file.

[llama.cpp](https://github.com/ggerganov/llama.cpp) project uses predefined pre-tokenizer outputs for the fixed string for the most popular BPE models to identify current pre-tokenizer.

If convertation fails with the following exception:

```log
Runtime error: BPE pre-tokenizer was not recognized - update get_vocab_base_pre()
```

you may need to add the model to the `convert_hf_to_gguf_update.py` script and run it to update the main convertation script. Follow the instructions at the begginning of the `convert_hf_to_gguf_update.py`.

More information about this script you can find [here](https://github.com/ggerganov/llama.cpp/pull/6920).

## Example

Download small GPT2 model from [this repo](https://huggingface.co/openai-community/gpt2) to `tests` directory.

Convert the model to GGUF with:

```shell
python convert_hf_to_gguf.py tests/gpt2 --outfile tests/gpt2.gguf
```
