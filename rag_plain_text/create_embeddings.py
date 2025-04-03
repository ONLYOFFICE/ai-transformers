from datasets import Dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Create a vector database for RAG from specified text document")
    parser.add_argument("model_id", help="embedding model ID from Huggingface hub or the PATH to the directory with embedding model")
    parser.add_argument("text_file", help="path to the text file (must be in UTF-8 encoding)")
    parser.add_argument("-s", "--chunk-size", dest="chunk_size", type=int, help="the maximum number of characters in each chunk (default: 1000)", default=1000)
    parser.add_argument("-v", "--chunk-overlap", dest="chunk_overlap", type=int, help="number of characters to overlap between chunks (default: 200)", default=200)
    parser.add_argument("--sep", dest="separator", help="the character to split on (defaults to newline)", default="\n")
    parser.add_argument("-o", "--out", dest="out_dir", help="path to the output directory containing dataset with embeddings (defaults to './out')", default="./out")
    parser.add_argument("-p", "--prompt", dest="passage_prompt", help="prompt to prefix every passage (no prefix prompt by default)", default=None)
    args = parser.parse_args()
    model_id = args.model_id
    text_file = args.text_file
    out_dir = args.out_dir
    passage_prompt = args.passage_prompt
    # read and split the file
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()
    splitter = CharacterTextSplitter(chunk_overlap=args.chunk_overlap, chunk_size=args.chunk_size, separator=args.separator)
    chunks = splitter.split_text(text)
    print(f"info: the text was split into {len(chunks)} chunks.")
    # form dataset
    dataset = Dataset.from_dict({"content": chunks})
    # load model
    model = SentenceTransformer(model_id)
    # define embedding function
    def embed_function(example):
        return {"embeddings": model.encode(example['content'], prompt=passage_prompt, normalize_embeddings=True).tolist()}
    # create embeddings in columng "embeddings"
    dataset = dataset.map(embed_function, batched=False)
    # save dataset to directory
    dataset.save_to_disk(out_dir)

if __name__ == "__main__":
    main()
