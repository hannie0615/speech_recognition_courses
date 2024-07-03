#!/usr/bin/python3

# Refer to https://huggingface.co/docs/transformers/en/tasks/asr

from transformers import AutoModelForCTC, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
import numpy as np

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

#gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")
#dataset = load_dataset("google/speech_commands", "v0.02")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.remove_columns(
    ["english_transcription", "intent_class", "lang_id"])


print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")
device = torch.device("mps")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = evaluate.load("wer")
    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def prepare_dataset(batch):
    audio = batch["audio"]

    batch = processor(audio["array"],     # add device,
                      sampling_rate=audio["sampling_rate"],
                      text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch


def uppercase(example):
    return {"transcription": example["transcription"].upper()}


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{
            "input_values": feature["input_values"][0]
        } for feature in features]
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]

        batch = self.processor.pad(input_features,
                                   padding=self.padding,
                                   return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features,
                                          padding=self.padding,
                                          return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels.to(device)     # add device

        return batch


dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
dataset = dataset.map(uppercase)
dataset = dataset.map(prepare_dataset,
                      remove_columns=dataset.column_names["train"],
                      num_proc=4)


data_collator = DataCollatorCTCWithPadding(processor=processor,
                                           padding="longest")

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id)

#import pdb;
#pdb.set_trace()
#for i, data in enumerate(dataset):
#    print(i)
#    print(data)
#    print(data.keys())
#    break

# asr_mind_model01: 16 batch, 1e-4
# asr_mind_model01: 16 batch, 3e-4

training_args = TrainingArguments(
    output_dir="asr_mind_model_00",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    max_steps=20000,
    gradient_checkpointing=True,
    fp16=False, # mps 때문에 수정
    group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    save_steps=1000,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
# trainer.fit()