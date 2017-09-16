import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time



# Converts to mono
def toMono(stereo):
    if (len(stereo[0]) == 2):
        newData = np.zeros((len(stereo), 1), dtype=np.int16)
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


def getWaveform():
    rate, data = wav.read('Music Sample/ShortRabbit.wav', True)
    data = toMono(data)
    t = np.arange(len(data[:, 0])) * 1.0 / rate
    return data


def getCount():
    return count

def addCount(num):
    count = getCount() + num

def checkError(new, master, time):
    print('User:' + str(new))
    print('Master:' + str(master[time]))
    return master[time] - 10*new

#take max and alter data
def callback(inData, frame_count, time_info, status):
    audioData = np.fromstring(inData, dtype=np.int16)
    print('Error:' + str(checkError(audioData, data, getCount())))
    addCount(1)
    return (audioData, pyaudio.paContinue)


count = 0
data = getWaveform()

p = pyaudio.PyAudio()
buffer = 1024
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, output=True, frames_per_buffer=1, stream_callback= callback)

start = time.time()
elapsed = time.time() - start
while stream.is_active() & (elapsed < len(data)/44100):
    time.sleep(0.1)
    elapsed = time.time() - start


stream.stop_stream()
stream.close()
p.terminate()





# np.asarray(data)
# t = np.arange(len(data[:, 0])) * 1.0 / 44100
# plt.plot(t, data, 'r--')

# getAndDrawWaveform()








