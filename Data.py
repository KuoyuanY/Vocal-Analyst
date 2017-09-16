import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
from Waves import getAndDrawWaveform

def compareWavesGraph(fileMaster, fileUser):
    getAndDrawWaveform(fileMaster, 'r--', 1)
    getAndDrawWaveform(fileUser, 'b--', 2)



compareWavesGraph('Music Sample/White Rabbit.wav', 'Music Sample/White Rabbit.wav')