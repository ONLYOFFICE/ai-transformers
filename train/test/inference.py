from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import argparse

def generate(model, tokenizer, device, system_prompt, user_prompt):
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
    out = model.generate(tokenized['input_ids'], attention_mask=tokenized['attention_mask'], max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][len(tokenized[0]):-1])
    return response

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run an interfere loop on selected model")
    parser.add_argument("model_id", help="model ID from Huggingface hub or the PATH to the directory with LLM")
    parser.add_argument("dataset_file", help="path to the JSONL dataset that is needed for acquiring the system prompt (should be in chat-template format)")
    args = parser.parse_args()
    model_id = args.model_id
    dataset_file = args.dataset_file
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device)
    # load dataset to get system prompt
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    system_prompt = dataset[0]['messages'][0]['content']
    # run LLM inference loop
    print("Running LLM inference loop. Press Ctrl+C to exit.")
    try:
        while True:
            user_input = input("> ")
            if not user_input.strip():
                print("Prompt cannot be empty. Try again.")
                continue

            response = generate(model, tokenizer, device, system_prompt, user_input)
            print(f": {response}")
    except KeyboardInterrupt:
        print("\nExiting inference loop.")

if __name__ == "__main__":
    main()
