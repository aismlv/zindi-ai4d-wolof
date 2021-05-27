import json
import re
import subprocess
from pathlib import Path

import datasets
import pandas as pd
import torchaudio
from sklearn.model_selection import KFold

save_path = Path("data/")
preprocessing_num_workers = 2

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

# split by labels
labels = train["transcription"].unique()
kf = KFold(n_splits=10, shuffle=True, random_state=2).split(labels)

train_labels, valid_labels = next(kf)
train_labels, valid_labels = labels[train_labels], labels[valid_labels]

train_idx = train["transcription"].isin(train_labels)
valid_idx = train["transcription"].isin(valid_labels)

train, valid = train[train_idx].set_index("ID"), train[valid_idx].set_index("ID")

train_dataset = datasets.Dataset.from_pandas(train)
valid_dataset = datasets.Dataset.from_pandas(valid)
test_dataset = datasets.Dataset.from_pandas(test)


# process text
chars_to_ignore = ["'", "\-", ",", "(", ")", "’", '"']
chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'


def process_text(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, "", batch["transcription"])
    batch["text"] = batch["text"].lower() + " "
    batch["text"] = batch["text"]
    return batch


train_dataset = train_dataset.map(process_text)
valid_dataset = valid_dataset.map(process_text)

text_col = "text"

all_text = " ".join(train_dataset[text_col] + valid_dataset[text_col])

vocab_list = list(set(all_text))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open(save_path / "vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)


# process audio
def process_speech_to_array(batch):
    path = save_path / f"clips/{batch['ID']}.mp3"
    speech_array, sr = torchaudio.load(path)
    batch["speech"] = (
        torchaudio.transforms.Resample(sr, 16_000)(speech_array).squeeze().numpy()
    )
    batch["duration"] = len(batch["speech"]) / 16_000
    batch["sampling_rate"] = sr
    return batch


train_dataset = train_dataset.map(
    process_speech_to_array,
    num_proc=preprocessing_num_workers,
)

valid_dataset = valid_dataset.map(
    process_speech_to_array,
    num_proc=preprocessing_num_workers,
)

test_dataset = test_dataset.map(
    process_speech_to_array,
    num_proc=preprocessing_num_workers,
)


# save
train_dataset.save_to_disk(f"{save_path}/train.dataset")
valid_dataset.save_to_disk(f"{save_path}/valid.dataset")
test_dataset.save_to_disk(f"{save_path}/test.dataset")


# create language model
all_lines = (
    train["transcription"]
    .str.lower()
    .apply(lambda x: re.sub(chars_to_ignore_regex, "", x))
    .tolist()
)


text_input = "/content/zindi-ai4d-wolof/temp/kenlm-input.txt"
kenlm_path = "/tmp/kenlm/build/bin/"
lm_output_path = "/content/zindi-ai4d-wolof/temp/lm.arpa"

with open("/content/zindi-ai4d-wolof/temp/kenlm-input.txt", "w") as f:
    for line in all_lines:
        f.write(line + "\n")

subprocess.run(
    f"cat {text_input} | {kenlm_path}lmplz -o 3 > {lm_output_path}", shell=True
)
subprocess.run(
    f"{kenlm_path}/build_binary {lm_output_path} {lm_output_path}.bin", shell=True
)
