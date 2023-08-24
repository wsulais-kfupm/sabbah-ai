from faster_whisper import WhisperModel



model = WhisperModel("medium", device="cuda")

import pandas as pd
import datasets

root = "/ibex/user/sulaisw/sada/"

def get_dataset():
    df = pd.read_csv(root + "test.csv")
    #display(df.Speaker.unique())
    #display(df)
    df["audio"] = root + df["FileName"]
    ds = datasets.Dataset.from_pandas(df).cast_column("audio", datasets.Audio())
    return ds

def crop_audio(batch):
    sr = batch["audio"]["sampling_rate"]
    start = int(batch["SegmentStart"] * sr)
    end = int(batch["SegmentEnd"] * sr)
    batch["audio"]["array"] = batch["audio"]["array"][start:end]
    return batch


import evaluate
import torch

# model = model.to("mps")
def map_to_pred(batch):
    batch = crop_audio(batch)
    audio = batch["audio"]
    input_features = audio["array"]
    batch["reference"] = batch["ProcessedText"]

    segments, info = model.transcribe(input_features)
    batch["prediction"] = "".join(map(lambda x: x.text, segments)).strip()
    return batch

def maps(batch):
    audio = batch["audio"]
    input_features = audio["array"]
    batch["reference"] = batch["labels"]

    segments, info = model.transcribe(input_features)
    batch["prediction"] = "".join(map(lambda x: x.text, segments)).strip()
    return batch

#ds = get_dataset()
ds = datasets.load_from_disk("/ibex/user/sulaisw/whisper/sada_ar_whisper_medium_train")["test"]
ds1 = datasets.load_from_disk("/ibex/user/sulaisw/sada_hugging")["test"]
result = ds.map(maps)
result.save_to_disk("test.hf")

#ds["train"]["ProcessedText"]

from evaluate import load
wer = load("wer")
print("WER:", 100 * wer.compute(references=result["ProcessedText"], predictions=result["prediction"]))

cer = load("cer")
print("CER:", 100 * cer.compute(references=result["ProcessedText"], predictions=result["prediction"]))
