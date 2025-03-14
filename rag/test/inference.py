from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset
import argparse
from typing import Optional

rag_prompt = """
You have access to retrieved data that provide relevant information to answer user questions accurately.

Follow these guidelines when responding:
 1. **Use the retrieved data as your primary and only source of truth.**
 2. **Only generate answers based on retrieved information.** If the data do not contain relevant information, say: "I couldn't find relevant information."
 3. **Incorporate the information naturally into your responses** without mentioning that it was retrieved.
 4. **Do not make up facts or speculate.**
 5. **Summarize and synthesize retrieved information in a clear and concise manner.**
 6. **If multiple retrieved sources provide conflicting information, acknowledge the discrepancy.**

Always aim for clarity, factual accuracy, and helpfulness.
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
    # get initial system prompt
    system_prompt = dataset[0]['messages'][0]['content']
    # retrieve data from index
    context = dataset.get_nearest_examples("embeddings", embedding_model.encode(user_prompt, prompt=query_prompt, normalize_embeddings=True), k=top_k)
    # print retrieved examples
    if show_all:
        print("<context>")
        print("Score:\tExample:")
        for i, example in enumerate(context.examples['messages']):
            print(f"{context.scores[i]:6.2f}\t{example[1]['content']}")
        print("</context>")
    # add retrieved data to system prompt
    system_prompt += f"\n{rag_prompt}"
    system_prompt += f"\n### Retrieved Information:\n"
    for example in context.examples['messages']:
        system_prompt += f" - {example[1]['content']}\n"
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
            f"### System:\n{system_prompt}\n\n"
            f"### Instruction:\n{user_prompt}\n\n"
            f"### Response:\n"
        ]
        tokenized = tokenizer(prompt_formatted, truncation=True, return_tensors="pt").to(device)
    out = model.generate(tokenized['input_ids'], attention_mask=tokenized['attention_mask'], max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][len(tokenized[0]):], skip_special_tokens=True)
    return response

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run an inference loop on selected model with RAG", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_id", help="model ID from Huggingface hub or the PATH to the directory with LLM")
    parser.add_argument("dataset_file", help="path to the JSONL dataset that is needed for acquiring the system prompt (should be in chat-template format)")
    parser.add_argument("-i", "--index-file", dest="index_file", help="path to the FAISS index file", required=True)
    parser.add_argument("-e", "--embedding-model", dest="embedding_model_id", help="embedding model ID from Huggingface hub or the PATH to the directory with the model. It should be the same model as the embedding model that has been used to create FAISS index", required=True)
    parser.add_argument("-p", "--prompt", dest="query_prompt", help="prompt to prefix every query when retrieving data", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, help="number of rows that will be retrieved from FAISS index and provided to chat model as context for user prompt", default=3)
    parser.add_argument("--show-all", dest="show_all", action="store_true", help="show retrieved data for each query")
    args = parser.parse_args()
    model_id = args.model_id
    dataset_file = args.dataset_file
    index_file = args.index_file
    embedding_model_id = args.embedding_model_id
    query_prompt = args.query_prompt
    top_k = args.top_k
    show_all = args.show_all
    # load tokenizer and chat model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    # load embedding model
    embedding_model = SentenceTransformer(embedding_model_id)
    # load dataset
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    # load FAISS index
    dataset.load_faiss_index("embeddings", file=index_file)
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
