import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pyaudio

# Converts to mono
def toMono(stereo):
    if (len(stereo[0]) == 2):
        newData = np.zeros((len(stereo), 1), dtype=np.int32)
        iterate = 0
        for values in stereo:
            newData[iterate] = ((values[0] + values[1]) / 2)
            iterate += 1
    elif (len(stereo[0]) == 1):
        # Already mono
        newData = stereo
    else:
        raise TypeError
    return newData


def getAndDrawWaveform():
    rate, data = wav.read('Music Sample/ShortRabbit.wav', True)
    data = toMono(data)
    t = np.arange(len(data[:, 0])) * 1.0 / rate
    plt.plot(t, data, 'r--')
    plt.show()
    return data

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt32, channels=1, rate=44100, frames_per_buffer=1024)
getAndDrawWaveform()








