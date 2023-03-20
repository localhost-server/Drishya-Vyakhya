import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import numpy as np

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
def tacotron():
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")#,run_opts={"device":"cuda"})
    return tacotron2

def hifigan():
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
    return hifi_gan