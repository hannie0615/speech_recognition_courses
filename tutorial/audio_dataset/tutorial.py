#!/usr/bin/python3

# Refer to https://huggingface.co/docs/transformers/en/tasks/asr

from transformers import AutoModelForCTC, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

#gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")
#dataset = load_dataset("google/speech_commands", "v0.02")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.remove_columns(
    ["english_transcription", "intent_class", "lang_id"])


def prepare_dataset(batch):
    audio = batch["audio"]

    batch = processor(audio["array"],
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

        batch["labels"] = labels

        return batch


dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
dataset = dataset.map(uppercase)

dataset = dataset.map(prepare_dataset,
                      remove_columns=dataset.column_names["train"],
                      num_proc=4)


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


training_args = TrainingArguments(
    output_dir="my_awesome_asr_mind_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
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

