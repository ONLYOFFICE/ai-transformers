# Model converter

Converts `.safetensors` models to `.gguf`. Taken as is from [llama.cpp](https://github.com/ggerganov/llama.cpp).

> **NOTE:** before using the script install all requirements with:

```shell
pip install -r requirements.txt
```
## Usage

```shell
python convert_hf_to_gguf.py model --outfile OUTFILE
```
 - `model` - the directory containing model file
 - `OUTFILE` - output model name with `.gguf` extension

The script provides variety of settings including quantization via `--outtype` flag. Run:

```shell
python convert_hf_to_gguf.py -h
```
to see all of the options or head to official documentation in llama.cpp repo.

## Example

Download small GPT2 model from [here](https://huggingface.co/openai-community/gpt2) to `tests` directory.

Convert the model to GGUF with:

```shell
python convert_hf_to_gguf.py tests/gpt2 --outfile tests/gpt2.gguf
```
