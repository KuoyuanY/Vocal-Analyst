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


def getAndDrawWaveform(file, color, figure):
    rate, data = wav.read(file, True)
    data = toMono(data)
    t = np.arange(len(data[:, 0])) * 1.0 / rate
    plt.figure(figure)
    plt.plot(t, data, color)
    plt.show()
    return data


def getWaveform():
    rate, data = wav.read('Music Sample/White Rabbit.wav', True)
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
    print('Error:' + str(checkError(audioData, stream().data, getCount())))
    addCount(1)
    return (audioData, pyaudio.paContinue)


def stream():
    count = 0
    data = getWaveform()

    p = pyaudio.PyAudio()
    buffer = 1024
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, output=True, frames_per_buffer=1, stream_callback= callback)

    start = time.time()
    elapsed = time.time() - start
    while stream.is_active() & (elapsed < len(data)/44100):
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    p.terminate()

# getAndDrawWaveform('Music Sample/White Rabbit.wav', 'r--')
# getAndDrawWaveform('Music Sample/White Rabbit.wav', 'b--')


# np.asarray(data)
# t = np.arange(len(data[:, 0])) * 1.0 / 44100
# plt.plot(t, data, 'r--')

# getAndDrawWaveform()
def compareWavesGraph(fileMaster, fileUser):
    getAndDrawWaveform(fileMaster, 'r--', 1)
    getAndDrawWaveform(fileUser, 'b--', 2)
    getWaveform()


compareWavesGraph('Music Sample/ShortRabbit.wav', 'Music Sample/ShortRabbit.wav')







