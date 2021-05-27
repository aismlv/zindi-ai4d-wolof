import argparse
import re
from pathlib import Path

import ctcdecode
import datasets
import numpy as np
import pandas as pd
import torch
from Levenshtein import distance
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main(args):
    data_dir = Path("data/")
    train = pd.read_csv(data_dir / "Train.csv").drop("ID", axis=1)

    def evaluate_batch(batch):
        inputs = processor(
            batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = model(
                inputs.input_values.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
            ).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        batch["logits"] = logits.cpu().numpy()
        return batch

    def evaluate(dataset, batch_size):
        dataset.sort("duration")
        result = dataset.map(evaluate_batch, batched=True, batch_size=batch_size)
        return result

    # predict
    if args.do_expand:
        model_name = "step-1"
    else:
        model_name = "step-2"

    model_path = f"/content/output/{model_name}"

    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to("cuda")

    test_dataset = datasets.load_from_disk(data_dir / "test.dataset")
    test_preds = evaluate(test_dataset, batch_size=1)
    test_df = test_preds.to_pandas().drop(["speech"], axis=1)

    # match labels
    chars_to_ignore = ["'", "\-", ",", "(", ")", "’", '"']
    chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'

    def process_text(s):
        s = re.sub(chars_to_ignore_regex, "", s)
        return s.lower().strip()

    train["labels"] = train["transcription"].apply(process_text)
    labels = train["labels"].unique().tolist()

    # create dict matching processed labels to original text field
    df = train.loc[
        ~train[["labels", "transcription"]].duplicated(), ["labels", "transcription"]
    ]
    labels2inputs = {l: t for _, l, t in df.itertuples()}

    # find 1NN
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        s = row["pred_strings"]
        distances = np.array([distance(s, label) for label in labels])

        match_id = np.argmin(distances)
        min_dist = distances[match_id]
        test_df.at[idx, "neighbour"] = labels[match_id]
        test_df.at[idx, "distance"] = min_dist
        test_df.at[idx, "n_matches"] = (distances == min_dist).sum()

    # fill-in predictions
    cutoff = 7
    test_df.loc[test_df["distance"] <= cutoff, "prediction"] = test_df["neighbour"].map(
        labels2inputs
    )
    test_df.loc[test_df["distance"] > cutoff, "prediction"] = test_df["pred_strings"]

    if args.do_expand:
        test_dataset = datasets.load_from_disk("data/test.dataset")

        def add_neighbour(batch):
            row = test_df.loc[batch["ID"]]
            batch["transcription"] = row["prediction"]
            batch["distance"] = row["distance"]
            return batch

        test_dataset = test_dataset.map(add_neighbour)
        test_dataset = test_dataset.filter(lambda batch: batch["distance"] <= 10)

        def process_text(batch):
            batch["text"] = re.sub(chars_to_ignore_regex, "", batch["transcription"])
            batch["text"] = batch["text"].lower() + " "
            batch["text"] = batch["text"]
            return batch

        test_dataset = test_dataset.map(process_text)
        test_dataset = test_dataset.remove_columns("distance")

        train_dataset = datasets.load_from_disk("data/train.dataset")
        valid_dataset = datasets.load_from_disk("data/valid.dataset")
        train_dataset = datasets.concatenate_datasets(
            [train_dataset, test_dataset, valid_dataset]
        )
        train_dataset.save_to_disk("data/train.dataset.expanded")
        return

    # decode remaining with a language model
    target_dictionary = {
        v: k for k, v in processor.tokenizer.get_vocab().items() if v < 37
    }
    target_dictionary[0] = " "
    target_dictionary[36] = "_"
    target_dictionary = [target_dictionary[i] for i in range(37)]

    lm_output_path = "/content/zindi-ai4d-wolof/temp/lm.arpa"

    word_lm_scorer = ctcdecode.WordKenLMScorer(lm_output_path, 2.5, 0.0)
    decoder = ctcdecode.BeamSearchDecoder(
        target_dictionary,
        num_workers=2,
        beam_width=64,
        scorers=[word_lm_scorer],
        cutoff_prob=np.log(0.000001),
        cutoff_top_n=40,
    )

    test_to_decode = test_df.loc[test_df["distance"] > cutoff]
    for idx, row in tqdm(test_to_decode.iterrows()):
        test_df.loc[idx, "prediction"] = decoder.decode(row["logits"])

    # create submission file
    test_df = test_df.set_index("ID")
    sub = pd.read_csv(data_dir / "SampleSubmission.csv").set_index("ID")
    sub["transcription"] = test_df["prediction"]
    sub.to_csv("submission.csv", index=True, header=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse stage")
    parser.add_argument(
        "--do-expand",
        default=False,
        action="store_true",
        help="True if want to create expanded train dataset",
    )
    args = parser.parse_args()
    main(args)
