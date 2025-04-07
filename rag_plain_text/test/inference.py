from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import argparse
from typing import Optional

rag_prompt = """
You have access to the following context below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Your answer should be based only on the provided context. Do not use any other data for your answer.
"""

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    embedding_model: SentenceTransformer,
    dataset: Dataset,
    user_prompt: str,
    query_prompt: Optional[str],
    top_k: int,
    show_all: bool
) -> str:
    # default system prompt
    system_prompt = "You are a helpful assistant."
    # retrieve data from index
    context = dataset.get_nearest_examples("embeddings", embedding_model.encode(user_prompt, prompt=query_prompt, normalize_embeddings=True), k=top_k)
    # print retrieved examples
    if show_all:
        print("<context>")
        for i, example in enumerate(context.examples['content']):
            print(f"<entry score:{context.scores[i]:6.2f}>")
            print(example)
            print("</entry>\n")
        print("</context>")
    # format context as a string
    context_str = ""
    for example in context.examples['content']:
        context_str += f"{example}\n\n"
    context_str = context_str[:-2]
    # add retrieved data to system prompt
    system_prompt += rag_prompt.format(context_str=context_str)
    # generate a response
    device = model.device
    if tokenizer.chat_template:
        prompt_formatted = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tokenized = tokenizer.apply_chat_template(prompt_formatted, return_dict=True, truncation=True, add_generation_prompt=True, return_tensors="pt").to(device)
    else:
        prompt_formatted = [
            f"{system_prompt}\n"
            f"Query: {user_prompt}\n"
            f"Answer:\n"
        ]
        tokenized = tokenizer(prompt_formatted, truncation=True, return_tensors="pt").to(device)
    out = model.generate(tokenized['input_ids'], attention_mask=tokenized['attention_mask'], max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
    response = tokenizer.decode(out[0][len(tokenized[0]):], skip_special_tokens=True)
    return response

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run an inference loop on selected model with RAG", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_id", help="model ID from Huggingface hub or the PATH to the directory with LLM")
    parser.add_argument("embeddings", help="path to the dataset with embeddings")
    parser.add_argument("-e", "--embedding-model", dest="embedding_model_id", help="embedding model ID from Huggingface hub or the PATH to the directory with the model. It should be the same model as the embedding model that has been used to create FAISS index", required=True)
    parser.add_argument("-p", "--prompt", dest="query_prompt", help="prompt to prefix every query when retrieving data", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, help="number of rows that will be retrieved from FAISS index and provided to chat model as context for user prompt", default=3)
    parser.add_argument("--show-all", dest="show_all", action="store_true", help="show retrieved data for each query")
    args = parser.parse_args()
    model_id = args.model_id
    dataset_dir = args.embeddings
    embedding_model_id = args.embedding_model_id
    query_prompt = args.query_prompt
    top_k = args.top_k
    show_all = args.show_all
    # load tokenizer and chat model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    # print chat template info
    if tokenizer.chat_template:
        print("info: chat template will be used")
    # load embedding model
    embedding_model = SentenceTransformer(embedding_model_id)
    # load dataset
    dataset = Dataset.load_from_disk(dataset_dir)
    # load FAISS index
    dataset.add_faiss_index("embeddings")
    # run LLM inference loop
    print("Running LLM inference loop. Press Ctrl+C to exit.")
    try:
        while True:
            user_input = input("> ")
            if not user_input.strip():
                print("Prompt cannot be empty. Try again.")
                continue

            response = generate(model, tokenizer, embedding_model, dataset, user_input, query_prompt, top_k, show_all)
            print(f": {response}")
    except KeyboardInterrupt:
        print("\nExiting inference loop.")

if __name__ == "__main__":
    main()
