import os

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment


print("Imports done")

work_dir = './train_dir'

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

train_dataset = MsDataset.load(
        'Libri2Mix_8k', split='train').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)

print("\nTrain dataset downloaded")

eval_dataset = MsDataset.load(
        'Libri2Mix_8k', split='validation').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)

print("Eval Dataset Downloaded")

kwargs = dict(
    model='damo/speech_mossformer_separation_temporal_8k',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir)

print("\nBuilt Model")

trainer = build_trainer(
    Trainers.speech_separation, default_args=kwargs)

print("\nBuilt Trainer")

trainer.train()
