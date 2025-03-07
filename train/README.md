# Fine-tuning LLM

**Fine-tuning** - is the process of training the pretrained AI model on a dataset specific to your task. For this purposees the _finetune.py_ Python script was developed. It allows you to run fine-tuning on different LLMs on a specified dataset automatically.

# Installing dependencies

> NOTE: It is highly recommended to install all dependencies inside a virtual Python environment.

First of all, you need to figure out what CUDA version is supported by your GPU (if it is avaiable at all). Run the following command to check info about available GPUs:

```shell
nvidia-smi
```

Then using [PyTorch](https://pytorch.org/get-started/locally/#start-locally) site acquire the URL of the appropriate PyTorch package. Next, change url in [requirements.txt](requirements.txt) and run

```shell
pip install -r requirements.txt
```

to install all dependencies.

# Usage

You can now run script with:

```shell
python finetuning.py model_id dataset_file
```

where
 - `model_id` - path to the local directory containing LLM or model identificator from huggingface hub.
 - `dataset_file` - path to the local dataset file. Dataset should be in _chat template_ format and formatted as JSONL file. One row of chat template dataset should look like:

   ```json
   "messages": [
       {
           "role": "system",
           "content": "SYSTEM_PROMPT"
       },
       {
           "role": "user",
           "content": "USER_INPUT"
       },
       {
           "role": "assistant",
           "content": "MODEL_RESPONSE"
       }
   ]
   ```
   The dataset ***must***  contain `message`. Other columns are optional and will be ignored.

To see all available options run:

```shell
python finetuning.py -h
```

These options are:
 - `--out OUT_DIR` - output dir where fine-tuned model will be saved. By default it is `./out`.
 - `--ratio DATA_RATIO` - the value in range `(0.0, 1.0]` which sets the fraction of number of all rows from dataset to be used during training. Defaults to `1.0` which means that the whole dataset is going to be used.
 - `--steps STEPS` - number of the steps before every evaluation and checkpoint saving. Defaults to `500`.
 - `--validate` - if this option is specified, the script will run some additional validations before the fine-tuning process to make sure it goes through as expected.

# Example

There is a sample dataset [doctor.jsonl](test/datasets/doctor.jsonl) that represents the requests from patient and corresponding doctor's answers about the treatment and diagnosis. Let's run fine-tuning on the model [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (you may need to request access to this model).

```shell
python finetune.py meta-llama/Llama-3.2-1B-Instruct ./test/datasets/doctor.jsonl --out tests/out --steps 200
```

During fine-tuning process you will see the overall progress and some metrics: loss, evaluation loss, evaluation runtime, epoch number etc. At the end of the fine-tuning the program will choose the checkpoint with least evaluation loss and save it in the output directory.

# Testing

To test the fine-tuned model run the script _test/inference.py_ which will run the inference loop on the specified model:

```shell
python ./test/inference.py ./tests/out/ ./test/datasets/doctor.jsonl
```

> NOTE: Context is not saved during the LLM inference, which means every new response will be generated with no information about previous messages.

If you want to see the special tokens that model generates (such as `<|endoftext|>`, `<|im_end|>`, etc.) provide additional `--show-all` option:

```shell
python ./test/inference.py ./tests/out/ ./test/datasets/doctor.jsonl --show-all
```

Just type your requests in console and see what LLM will generate as a response.