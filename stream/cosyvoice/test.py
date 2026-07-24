import sys
import torchaudio
#sys.path.append('cosyvoice')
#sys.path.append('third_party/Matcha-TTS')

import os
path = 'third_party/Matcha-TTS'
print(os.path.abspath(path))

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel as CosyVoice2

def tts_model(text, audio_data):
    model = CosyVoice2(model_dir='/home/ysr/.cache/modelscope/hub/iic/CosyVoice2-0.5B')
    for i, j in enumerate(model.inference_instruct2(text, '请说话语速自然有一定起伏波动<|endofprompt|>', audio_data, stream=True)):
        yield (24000, j['tts_speech'])


#audio_data, sample_rate = torchaudio.load("zero_shot_prompt.wav")
audio_data = "zero_shot_prompt.wav"

audio_out = tts_model("吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮", audio_data)

for audio in audio_out:
    sample_rate, data = audio
    print(sample_rate, data)
