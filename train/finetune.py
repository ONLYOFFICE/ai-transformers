from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
import argparse

# custom data collator that will automatically pad 'labels' with -100 to prevent loss calculation on non-desired values
class DataCollatorWithLabelsMasking(DataCollatorWithPadding):
    def __call__(self, features):
        labels = []
        for feature in features:
            labels.append(feature['labels'])
            del feature['labels']
        # dynamically pad 'input_ids' and 'attention_mask'
        collated = super().__call__(features)
        # mask system and user prompts in 'labels' with -100 to fine-tune the model only on desired answers
        # also mask all paddings, but preserve EOS token at the end because we want model to stop generating answer
        for i in range(len(features)):
            left_mask_length = len(features[i]['input_ids']) - len(labels[i])
            right_mask_length = len(collated['input_ids'][i]) - len(features[i]['input_ids'])
            labels[i] = [-100] * left_mask_length + labels[i] + [-100] * right_mask_length
        collated['labels'] = torch.tensor(labels, dtype=torch.long)
        return collated

    @staticmethod
    def validate(data_loader, tokenizer):
        # logger function
        def print_invalids(input_id, label):
            print(tokenizer.decode(input_id))
            print("---")
            print(tokenizer.decode(label[label > 0]))
        # validation
        print("DATA LOADER VALIDATION")
        for batch in data_loader:
            for i in range(len(batch['input_ids'])):
                for id_token, label_token in zip(batch['input_ids'][i], batch['labels'][i]):
                    if label_token == -100:
                        continue
                    if id_token != label_token:
                        print_invalids(batch['input_ids'][i], batch['labels'][i])
                        raise RuntimeError("VALIDATION IS UNSUCCESSFULL")
        print("VALIDATION IS SUCCESSFULL")

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune pretrained LLM on the specified dataset")
    parser.add_argument("model_id", help="model ID from Huggingface hub or the PATH to the directory with LLM")
    parser.add_argument("dataset_file", help="path to the JSONL dataset to train LLM (should be in chat-template format)")
    parser.add_argument("-o", "--out", dest="out_dir", help="path to the output directory that will contain fine-tuned model files (defaults to './out')", default="./out")
    parser.add_argument("-r", "--ratio", dest="data_ratio", help="value in range (0.0, 1.0] that represents the ratio of how many rows of dataset to use (defaults to '1.0')", default="1.0")
    parser.add_argument("-s", "--steps", dest="steps", help="number of steps between evaluations (defaults to '500')", default="500")
    parser.add_argument("-v", "--validate", dest="validate", action="store_true", help="wether to do some validations before training")
    args = parser.parse_args()
    model_id = args.model_id
    dataset_file = args.dataset_file
    out_dir = args.out_dir
    validate = args.validate
    # validate arguments
    try:
        data_ratio = float(args.data_ratio)
        if data_ratio <= 0.0 or data_ratio > 1.0:
            raise ValueError()
    except ValueError:
        print("data_ratio should be the number in range (0.0, 1.0]")
        exit(1)
    try:
        eval_steps = int(args.steps)
        if eval_steps <= 0:
            raise ValueError()
    except ValueError:
        print("steps should be the integer number greater than 0")
        exit(1)
    # load dataset
    dataset = load_dataset(path="json", data_files=dataset_file, split="train")
    if data_ratio < 1.0:
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * data_ratio)))
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    if tokenizer.padding_side == "left":
        raise RuntimeError("The tokenizer has defatult padding_side equal to 'left', which is not supported.")
    # define tokenizing function
    def tokenize_function(examples):
        if tokenizer.chat_template:
            # if chat template is supported, tokenize with special function
            tokenized = tokenizer.apply_chat_template(examples['messages'], return_dict=True, truncation=True, add_generation_prompt=False)
        else:
            # if chat template isn't supported, format prompts and tokenize as usual
            examples_formatted = [
                f"### System:\n{example[0]['content']}\n\n"
                f"### Instruction:\n{example[1]['content']}\n\n"
                f"### Response:\n{example[2]['content']}{tokenizer.eos_token}"
                for example in examples['messages']
            ]
            tokenized = tokenizer(examples_formatted, truncation=True)
        # create labels
        responds = [example[2]['content'] + tokenizer.eos_token for example in examples['messages']]
        tokenized['labels'] = tokenizer(responds, add_special_tokens=False, truncation=True)['input_ids']
        # validate labels
        for id, label in zip(tokenized['input_ids'], tokenized['labels']):
            # if label is NOT the suffix of input_id, raise an error
            if label != id[-len(label):]:
                id_suffix = tokenizer.decode(id[-len(label):])
                label_suffix = tokenizer.decode(label)
                raise RuntimeError(
                    f"labels can not be determined automatically. Check your dataset for excess whitespaces.\n"
                    f"'input_id' is: '{id_suffix}'\n"
                    f"'label'    is: '{label_suffix}'\n"
                )
        return tokenized
    # tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])
    # split to test and eval datasets
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(0.1).values()
    # define data collator
    data_collator = DataCollatorWithLabelsMasking(tokenizer)
    if validate:
        data_loader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
        DataCollatorWithLabelsMasking.validate(data_loader, tokenizer)
    # define training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,             # directory for saving the fine-tuned model
        overwrite_output_dir=True,      # overwrite previous outputs
        eval_strategy="steps",          # evaluate every N steps
        save_strategy="steps",          # same for saving model as checkpoint
        eval_steps=eval_steps,          # number of steps N before evaluation
        save_steps=eval_steps,          # same for saving model
        logging_steps=eval_steps,       # log loss
        save_total_limit=4,             # keep only last 4 checkpoints
        per_device_train_batch_size=8,  # value can be adjusted based on GPU memory
        per_device_eval_batch_size=8,   # same as training batch size
        learning_rate=5e-5,             # standard LR for fine-tuning
        weight_decay=0.01,              # helps prevent overfitting
        warmup_ratio=0.1,               # gradually increase LR
        lr_scheduler_type="cosine",     # cosine decay for LR
        num_train_epochs=3,             # 3-5 epochs for general fine-tuning
        bf16=True,                      # use mixed precision if available
        load_best_model_at_end=True,    # load best model based on eval metric
        metric_for_best_model="loss",   # use loss to track best model
    )
    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    # start training
    trainer.train()
    # log best checkpoint
    print(f"The model was loaded from the best checkpoint {trainer.state.best_model_checkpoint}")
    # save
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()