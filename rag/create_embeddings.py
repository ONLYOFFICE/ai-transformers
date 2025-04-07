from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Create a vector database for RAG from specified dataset")
    parser.add_argument("model_id", help="embedding model ID from Huggingface hub or the PATH to the directory with embedding model")
    parser.add_argument("dataset_file", help="path to the JSONL dataset (should be in chat-template format)")
    parser.add_argument("-o", "--out", dest="out_file", help="path to the output vector database file (defaults to './index.faiss')", default="./index.faiss")
    parser.add_argument("-p", "--prompt", dest="passage_prompt", help="prompt to prefix every passage (no prefix prompt by default)", default=None)
    args = parser.parse_args()
    model_id = args.model_id
    dataset_file = args.dataset_file
    out_file = args.out_file
    passage_prompt = args.passage_prompt
    # load dataset
    dataset = load_dataset(path="json", data_files=dataset_file, split="train")
    # load model
    model = SentenceTransformer(model_id)
    # define embedding function
    def embed_function(examples):
        # include only user content in embeddings
        batch = [example[1]['content'] for example in examples['messages']]
        return {"embeddings": model.encode(batch, prompt=passage_prompt, normalize_embeddings=True).tolist()}
    # create embeddings in columng "embeddings"
    dataset = dataset.map(embed_function, batched=True, batch_size=32)
    # add faiss index
    dataset.add_faiss_index("embeddings")
    # save index to file
    dataset.save_faiss_index("embeddings", file=out_file)

if __name__ == "__main__":
    main()
