#!/usr/bin/env python
# coding: utf-8

import datasets

# root = "/ibex/user/sulaisw/"
# ds = datasets.load_from_disk(root + "sada_hugging")
# # Ignore multi-speaker for now
# ds = ds.filter(lambda x: x != "More than 1 speaker اكثر من متحدث", input_columns="Speaker")
# 
# 
# # In[2]:
# 
# 
from transformers import WhisperProcessor

MODEL_SIZE="medium"
"""The size of the model used for the processor"""
MODEL=MODEL_SIZE
"""The size of the model. usu. the same as `MODEL_SIZE` except for large_v2 """

processor = WhisperProcessor.from_pretrained(
    f"openai/whisper-{MODEL_SIZE}", language="arabic", task="transcribe"
)

def make_dataset(processor) -> datasets.Dataset:
    root = "/ibex/user/sulaisw/"
    ds = datasets.load_from_disk(root + "sada_hugging")
    # Ignore multi-speaker for now
    ds = ds.filter(lambda x: x != "More than 1 speaker اكثر من متحدث", input_columns="Speaker")

    def prepare_dataset(example):
        audio = example["audio"]

        example = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["labels"],
        )

        return example

    ds1 = ds.filter(lambda x: x <= 30, input_columns="SegmentLength").rename_column("ProcessedText", "labels")
    print(ds1)
    print(ds1["train"].info)
    ds1 = ds1.map(prepare_dataset, num_proc=1, writer_batch_size=500)
    ds1 = ds1.filter(lambda x: x is not None and not any(map(lambda y: y is None, x)), input_columns="labels")
    ds1 = ds1.remove_columns( ['ShowName', 'FullFileLength', 'SegmentID', 'SegmentLength', 'SegmentStart', 'SegmentEnd', 'SpeakerAge', 'SpeakerGender', 'SpeakerDialect', 'Speaker', 'Environment', 'GroundTruthText'])
    return ds1
    ...

DATASET_DIR = f"sada_ar_whisper_{MODEL_SIZE}_train"
try:
    ds1 = datasets.load_from_disk(DATASET_DIR)
except:
    ds1 = make_dataset(processor)
    ds1.save_to_disk(DATASET_DIR)

# In[ ]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[ ]:

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


# In[ ]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{MODEL}")


# In[ ]:


from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="arabic", task="transcribe", use_cache=True
)


# In[ ]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=50,
    max_steps=10000,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# In[ ]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds1["train"],
    eval_dataset=ds1["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


# In[ ]:


trainer.train()

