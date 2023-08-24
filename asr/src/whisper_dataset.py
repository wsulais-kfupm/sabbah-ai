#!/usr/bin/env python
# coding: utf-8

import datasets

root = "/ibex/user/sulaisw/"
ds = datasets.load_from_disk(root + "sada_hugging")
# Ignore multi-speaker for now
ds = ds.filter(lambda x: x != "More than 1 speaker اكثر من متحدث", input_columns="Speaker")


# In[2]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large", language="arabic", task="transcribe"
)


# In[3]:


def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["labels"],
    )

    return example


# In[4]:


print(ds["train"])
print(ds["validation"])
print(ds["test"])


# In[ ]:


# HACK: column rename is necessary for map later
ds1 = ds.filter(lambda x: x <= 30, input_columns="SegmentLength").rename_column("ProcessedText", "labels")
print(ds1)
print(ds1["train"].info)
ds1 = ds1.map(prepare_dataset, num_proc=1, writer_batch_size=500)
ds1 = ds1.filter(lambda x: x is not None and not any(map(lambda y: y is None, x)), input_columns="labels")
ds1 = ds1.remove_columns( ['ShowName', 'FullFileLength', 'SegmentID', 'SegmentLength', 'SegmentStart', 'SegmentEnd', 'SpeakerAge', 'SpeakerGender', 'SpeakerDialect', 'Speaker', 'Environment', 'GroundTruthText'])
ds1.save_to_disk("sada_ar_whisper")
