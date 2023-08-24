import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input = "/ibex/user/shiekhmf/sada/overlaid/6k_v_SBA_1875_2-seg_202_900-210_890_6k_v_SBA_377_0-seg_132_140-140_130.wav"

VOCAL_SEP_MODEL = "/ibex/user/shiekhmf/MossFormer/train_dir/save/CKPT+2023-08-20+11-11-24+00"

separation = pipeline(Tasks.speech_separation, model=VOCAL_SEP_MODEL)

result = separation(input)

for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)

#ASR_MODEL = "/ibex/user/shiekhmf/MossFormer/train_dir/save/checkpoint-3000"
#asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL)

#files = ['output_spk0.wav', 'output_spk1.wav']
#transcriptions = map(asr_pipeline, files)
#for i, trans in enumerate(transcriptions):
#    print("Speaker {i}:", trans)
