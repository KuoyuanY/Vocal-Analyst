import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
import scipy.fftpack
from pylab import*



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
    rate, data = wav.read('Music Sample/ShortRabbit.wav', True)
    data = toMono(data)
    # for thingamabob in data:
    #     print(thingamabob)
    t = np.arange(len(data[:, 0])) * 1.0 / rate
    return data, rate


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




# compareWavesGraph('Music Sample/ShortRabbit.wav', 'Music Sample/ShortRabbit.wav')
# array, freq = getWaveform()

sampFreq, snd = wav.read('Music Sample/ShortRabbit.wav')
snd = snd / (2.**15)
s1 = snd[:, 0]
timeArray = arange(0, 475120, 1)
timeArray = timeArray / sampFreq
timeArray = timeArray * 1000

# plt.plot(timeArray, s1, color='k')
# ylabel('Amplitude')
# xlabel('Time (ms)')
# plt.show()

n = len(s1)
p = np.fft.fft(s1)

nUniquePts = int(ceil((n+1)/2.0))
p = p[0:nUniquePts]
p = abs(p)

p = p / float(n) # scale by the number of points so that
                 # the magnitude does not depend on the length
                 # of the signal or on its sampling frequency
p = p**2  # square it to get the power

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
if n % 2 > 0: # we've got odd number of points fft
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
plt.plot(freqArray/1000, 10*log10(p), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')
plt.show()

