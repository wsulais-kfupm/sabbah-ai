import numpy
import soundfile as sf
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input can be a URL or a local path
inputFile = '/ibex/user/shiekhmf/sada/overlaid-3/6k_v_SBA2_2009_3-seg_21_830-31_190_6k_v_SBA_1730_1-seg_107_730-117_090.wav'

data, rate = sf.read(inputFile)

resampled = librosa.resample(data, orig_sr=rate, target_sr=8000)

sf.write("./resampled.wav", resampled, 8000)


separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer_separation_temporal_8k'
)

result = separation("./resampled.wav")
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)


