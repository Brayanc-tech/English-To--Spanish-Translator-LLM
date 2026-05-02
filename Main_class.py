#Prepare a dataset for supervised fine-tuning of a language model. The dataset should consist of pairs of English sentences and their corresponding Spanish translations.
import json
import os
import requests

#import .json file english_spanish.json
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "english_spanish.json")

with open(json_path, "r") as f:
    data = json.load(f)
print("Number of entries in the dataset:", len(data))

#format input using alpaca format
formatted_data = []
for entry in data:
    english_sentence = entry["English"]
    spanish_sentence = entry["Spanish (Puerto Rican Dialect)"]
    formatted_entry = {
        "input": english_sentence,
        "output": spanish_sentence
    }
    formatted_data.append(formatted_entry)


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n### Instruction:\nTranslate the English sentence into Spanish with a Puerto Rican dialect."
    )
    input_text = f"\n\n### Input:\n{entry['input']}"
    return instruction_text + input_text

#prepare pytorch dataloaders into a training, validation and test set
train_portion = int(len(formatted_data) * 0.85)  # 85% for training
test_portion = int(len(formatted_data) * 0.1)    # 10% for testing
val_portion = len(formatted_data) - train_portion - test_portion  # Remaining 5% for validation

train_data = formatted_data[:train_portion]
test_data = formatted_data[train_portion:train_portion + test_portion]
val_data = formatted_data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

#organize data into training branches
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

print(custom_collate_draft_1(batch))
#create target tokenIDs for training 


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

#replace padding tokens with placeholders -100
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


#create data loaders


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print("Device:", device)

from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

#instantiate the dataset and loader

from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)



val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)



#load a pretrained LLM
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import download_and_load_gpt2, load_weights_into_gpt



BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

#Intruction finetuning the LLM with dataloss and training loop

from llms_from_scratch.ch05 import (
    calc_loss_loader,
    train_model_simple,
 )

#Added a checkpoint loading mechanism to avoid retraining the model from scratch every time the code is run. Therefore saving computational resources and time cosidering computer restrictions

model.to(device)
model_path = os.path.join(script_dir, "english_spanish_gpt2_finetuned.pth")
checkpoint_loaded = False

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    checkpoint_loaded = True
    print("Loaded fine-tuned model from:", model_path)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)



import time

num_epochs = 2
train_losses, val_losses, tokens_seen = [], [], []

if not checkpoint_loaded:
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(model.state_dict(), model_path)
    print("Saved fine-tuned model to:", model_path)



#inspecting the modeling loss 

from llms_from_scratch.ch05 import plot_losses

if train_losses:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
else:
    print("Skipped loss plot because the model was loaded from a checkpoint.")


#extracting responses

from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text
from tqdm import tqdm


def extract_response(generated_text, prompt):
    response_text = generated_text[len(prompt):].strip()
    response_text = response_text.replace("### Response:", "").strip()

    for stop_marker in ("\n\n###", "<|endoftext|>"):
        if stop_marker in response_text:
            response_text = response_text.split(stop_marker)[0].strip()

    return response_text


def translate_english_to_spanish(
    english_sentence,
    model,
    tokenizer,
    max_new_tokens=80,
    temperature=0.0,
    top_k=None
):
    model.eval()
    prompt = format_input({"input": english_sentence}) + "\n\n### Response:\n"

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=50256
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    return extract_response(generated_text, prompt)


torch.manual_seed(123)

for entry in test_data[:3]:
    response_text = translate_english_to_spanish(entry["input"], model, tokenizer)

    print(format_input(entry))
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text}")
    print("-------------------------------------")


#qualitative evaluatuion

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    test_data[i]["model_response"] = translate_english_to_spanish(
        entry["input"],
        model,
        tokenizer
    )


responses_path = os.path.join(script_dir, "english_spanish_with_responses.json")
with open(responses_path, "w") as file:
    json.dump(test_data, file, indent=4)

print("Saved translation responses to:", responses_path)


#scoring the responses


def normalize_translation(text):
    return " ".join(text.lower().strip().split())


def token_overlap_score(predicted, expected):
    predicted_tokens = set(normalize_translation(predicted).split())
    expected_tokens = set(normalize_translation(expected).split())

    if not expected_tokens:
        return 0

    return len(predicted_tokens & expected_tokens) / len(expected_tokens)


exact_matches = 0
overlap_scores = []

for entry in test_data:
    model_response = entry["model_response"]
    expected_response = entry["output"]

    if normalize_translation(model_response) == normalize_translation(expected_response):
        exact_matches += 1

    overlap_scores.append(token_overlap_score(model_response, expected_response))

exact_match_score = exact_matches / len(test_data) if test_data else 0
average_overlap_score = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

print(f"Exact-match translation score: {exact_match_score:.2%}")
print(f"Average token-overlap score: {average_overlap_score:.2%}")
print(f"Correct translations: {exact_matches}/{len(test_data)}")

