import datasets

root = "/ibex/user/sulaisw/sada/"
ds1 = datasets.load_dataset(root, data_dir=".", streaming=False)
ds1


def load_audio(row):
    row["FileName"] = (root + row["FileName"])
    return row


def crop_audio(batch):
    sr = batch["audio"]["sampling_rate"]
    start = int(batch["SegmentStart"] * sr)
    end = int(batch["SegmentEnd"] * sr)
    batch["audio"]["array"] = batch["audio"]["array"][start:end]
    return batch

ds12 = ds1.map(load_audio).rename_column("FileName", "audio")
it = iter(ds1["train"])
print(next(it))
info_features = datasets.Features({ "audio": datasets.Value(dtype='string'),
                                                "FileName": datasets.Value(dtype='string'),
                                                 "ShowName": datasets.Value(dtype='string'),
                                                 "FullFileLength": datasets.Value(dtype='float32'),
                                                 "SegmentID": datasets.Value(dtype='string'),
                                                 "SegmentLength": datasets.Value(dtype='float32'),
                                                 "SegmentStart": datasets.Value(dtype='float32'),
                                                 "SegmentEnd": datasets.Value(dtype='float32'),
                                                 "SpeakerAge": datasets.Value(dtype='string'),
                                                 "SpeakerGender": datasets.Value(dtype='string'),
                                                 "SpeakerDialect": datasets.Value(dtype='string'),
                                                 "Speaker": datasets.Value(dtype='string'),
                                                 "Environment": datasets.Value(dtype='string'),
                                                 "GroundTruthText": datasets.Value(dtype='string'),
                                                 "ProcessedText": datasets.Value(dtype='string'),
                                                })
#ds12["train"].info.features = info_features
#ds12["validation"].info.features = info_features
#ds12["test"].info.features = info_features
print(ds12["train"].info)
ds13 = ds12.cast_column("audio", datasets.Audio()).map(crop_audio, writer_batch_size=500)
#ds13 = ds12.cast_column("audio", datasets.Audio()).map(crop_audio)
ds13.save_to_disk("sada_ar")
