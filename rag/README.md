# RAG

**RAG** (Retrieval-Augmented Generation) - is a technique that enhances LLMs by retrieving relevant information from an external knowledge base before generating responses. This approach improves accuracy, reduces hallucinations, and enables the model to provide more contextually relevant answers.

The Python script `create_embeddings.py` creates vector database using [FAISS](https://github.com/facebookresearch/faiss) library.

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

> NOTE: GPU is not that necessary for RAG as for fine-tuning since it is only  used during embeddings creation and LLM inference.

# Usage

You can now run script with:

```shell
python create_embeddings.py model_id dataset_file
```

where
 - `model_id` - path to the local directory containing **embedding model** or model identificator from huggingface hub.
 - `dataset_file` - path to the local dataset file. Dataset should be in **chat template** format and formatted as JSONL file. One row of chat template dataset should look like:

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
python create_embeddings.py -h
```

These options are:
 - `--out OUT_FILE` - path to the output vector database file. By default it is `./index.faiss`.
 - `--prompt PASSAGE_PROMPT` - the prefix prompt for every passage. Some models are trained in the way that they expect prefixes when embedding and retrieving data. For example, passage prompts can be required to be prefixed with `"passage: "` and retreiving prompts with `"query: "`. See FAQ section for [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2#faq) embedding model as example. By default no prefix is passed when creating embeddings.

# Example

There is a sample dataset [cat-facts.jsonl](test/datasets/cat-facts.jsonl) that consists of a list of 150 facts about cats. To create embeddings from this dataset let's use the embedding model [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). According to the documentation, this model does not need prompt prefixes, so it's not necessary for providing value to `--prompt`.

```shell
python create_embeddings.py BAAI/bge-base-en-v1.5 ./test/datasets/cat-facts.jsonl --out tests/index.faiss
```

# Testing

To try this vector database out run the script *test/inference.py* which will run the inference loop with RAG on the specified LLM.

```shell
python test/inference.py mistralai/Mistral-7B-Instruct-v0.3 ./test/datasets/cat-facts.jsonl -i tests/index.faiss -e BAAI/bge-base-en-v1.5
```

here we run an inference on the LLM [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using the same embedding model, which is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), and passing paths to the original dataset and the vector database created previously.

> NOTE: you may only use the same embedding model that was used to create the vector database.

Some extra options are:
 - `--show-all` - shows retrieved data with scores that will be passed to LLM as a context to answer user query.
 - `--top-k TOP_K` - specifies how many rows of dataset will be retrieved and passed to LLM. Default value is `3`.
