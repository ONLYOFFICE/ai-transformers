# RAG on a text file

> NOTE: see [RAG](../rag/README.md) on a dataset to see how it works on a JSONL datsets.

To use **RAG** for retrieving data from a plain text file, the file must first be split into multiple chunks of partially overlapping text data. The overlapping decreases the chanse that the requested data will be split into different chunks. These chunks essentially form a dataset, thus all further steps for creating embeddings and using them are mostly the same as for regular [RAG](../rag/README.md).

The Python script `create_embeddings.py` creates directory with dataset which includes both split data and embeddings.

# Installing dependencies

The same as for regular [RAG](../rag/README.md).

# Usage

Run script with:

```shell
python create_embeddings.py model_id text_file
```

where
 - `model_id` - path to the local directory containing **embedding model** or model identificator from huggingface hub.
 - `text_file` - path to the local text file. Could be any file in UTF-8 encoding. It is recommended that file be structured for better splitting.

To see all available options run:

```shell
python create_embeddings.py -h
```

These options are:
 - `-s, --chunk-size CHUNK_SIZE` - the maximum number of characters in each chunk (default: 1000). The actual number of character in a chunk may exceed this value if the chunk can not be divided according to specified `separator`. Tweak this value according to your needs and the size of the original document.
 - `v, --chunk-overlap` - number of characters to overlap between chunks (default: 200). Usually this value should be around 20-30% of chunk size. Lower values mean lower dataset size and faster retrieving, but higher chance of loosing context due to one portion of information being split into different chunks. Higher values will make dataset bigger, lowering the chance of loosing context.
 - `--sep` - the character (or string) to split on (default: `\n`).
 - `-o, --out OUT_DIR` - path to the output directory. By default it is `./out`.
 - `-p, --prompt PASSAGE_PROMPT` - the prefix prompt for every passage. Some models are trained in the way that they expect prefixes when embedding and retrieving data. For example, passage prompts can be required to be prefixed with `"passage: "` and retreiving prompts with `"query: "`. See FAQ section for [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2#faq) embedding model as example. By default no prefix is passed when creating embeddings.

# Example

There is a sample text document [port_kembla.md](test/samples/port_kembla.md) which is an article from Wikipedia about Port Kembla city. To create dataset with embeddings from this file let's use the embedding model [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). According to the documentation, this model does not need prompt prefixes, so it's not necessary for providing value to `--prompt`.

```shell
python create_embeddings.py BAAI/bge-base-en-v1.5 ./test/samples/port_kembla.md --out tests/out
```

# Testing

To try the resulted dataset with embeddings and see RAG in action, run the script *test/inference.py* which will run the inference loop with RAG on the specified LLM.

```shell
python test/inference.py mistralai/Mistral-7B-Instruct-v0.3 tests/out -e BAAI/bge-base-en-v1.5
```

here we run an inference on the LLM [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using the same embedding model, which is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), and passing path to the directory with dataset and embeddings that were created previously.

> NOTE: you may only use the same embedding model that was used to create the embeddings.

Some extra options are:
 - `--show-all` - shows retrieved text chunks with scores.
 - `--top-k TOP_K` - specifies how many chunks will be retrieved and passed to LLM. Default value is `3`.
