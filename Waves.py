import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
import scipy.fftpack
import scipy.signal
from pylab import *
from scipy import signal


# Converts to mono
def toMono(stereo):
    if (len(stereo[0]) == 2):
        newData = np.zeros((len(stereo), 1), dtype=np.int32)
        iterate = 0
        for values in stereo:
            try:
                newData[iterate] = ((values[0] + values[1]) / 2)
                iterate += 1
            except RuntimeWarning:
                print('Added dummy value')
                newData[iterate] = [0]

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


def getWaveform(file):
    rate, data = wav.read(file, True)
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
    return master[time] - 10 * new


# take max and alter data
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
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, output=True, frames_per_buffer=1,
                    stream_callback=callback)

    start = time.time()
    elapsed = time.time() - start
    while stream.is_active() & (elapsed < len(data) / 44100):
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


def get_max_correlation(original, match):
    z = scipy.signal.fftconvolve(original, match[::-1])
    lags = np.arange(z.size) - (match.size - 1)
    return lags[np.argmax(np.abs(z))]


# compareWavesGraph('Music Sample/ShortRabbit.wav', 'Music Sample/ShortRabbit.wav')
# array, freq = getWaveform()
def testGraph(Full, Part):
    Freq, snd1 = wav.read(Full)
    Freq, snd2 = wav.read(Part)

    if (len(snd2) > len(snd1)):
        a = len(snd2) - len(snd1)
        snd2 = snd2[:-a]
    else:
        a = len(snd1) - len(snd2)
        snd1 = snd1[:-a]

    snd1 = snd1 / (2. ** 15)
    snd2 = snd2 / (2. ** 15)

    s1 = snd1[:, 0]
    s2 = snd2[:, 0]

    timeArray1 = arange(0, len(snd1), 1)
    timeArray1 = timeArray1 / Freq
    timeArray1 = timeArray1 * 1000

    timeArray2 = arange(0, len(snd2), 1)
    timeArray2 = timeArray2 / Freq
    timeArray2 = timeArray2 * 1000

    n1 = len(s1)
    n2 = len(s2)

    p1 = np.fft.fft(s1)
    p2 = np.fft.fft(s2)

    const1 = p1
    const2 = p2

    nUniquePts1 = int(ceil((n1 + 1) / 2.0))
    nUniquePts2 = int(ceil((n2 + 1) / 2.0))

    p1 = p1[0:nUniquePts1]
    p2 = p2[0:nUniquePts2]

    p1 = abs(p1)
    p2 = abs(p2)

    p1 = p1 / float(n1)
    p2 = p2 / float(n2)  # scale by the number of points so that
    # the magnitude does not depend on the length
    # of the signal or on its sampling frequency
    p1 = p1 ** 2  # square it to get the power
    p2 = p2 ** 2

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    # we've got odd number of points fft
    if n1 % 2 > 0:
        p1[1:len(p1)] = p1[1:len(p1)] * 2
    else:
        p1[1:len(p1) - 1] = p1[1:len(p1) - 1] * 2  # we've got even number of points fft

    if n2 % 2 > 0:
        p2[1:len(p2)] = p2[1:len(p2)] * 2
    else:
        p2[1:len(p2) - 1] = p2[1:len(p2) - 1] * 2

    freqArray1 = arange(0, nUniquePts1, 1.0) * (Freq / n1)
    freqArray2 = arange(0, nUniquePts2, 1.0) * (Freq / n2)

    plt.plot(freqArray1 / 1000, 10 * log10(p1), color='k')
    plt.plot(freqArray2 / 1000, 10 * log10(p2), color='g')

    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')
    plt.show()

    print('Ok, starting IFFT')
    array1 = np.fft.ifft(const1)
    # array1 = array1.astype('int16')
    print(array1)
    array2 = np.fft.ifft(const2)
    # array2 = array2.astype('int16')
    print(array2)
    final = array1 - array2
    print(np.float32(final))
    return np.float32(final)


def getExpectedVoice(Full, Part):
    Part, Freq = getWaveform(Part)
    Full, Freq = getWaveform(Full)

    # print(get_max_correlation(Full, Part))
    Part = Part[-get_max_correlation(Full, Part):]

    if (len(Part) > len(Full)):
        a = len(Part) - len(Full)
        Part = Part[:-a]
    else:
        a = len(Full) - len(Part)
        Full = Full[:-a]

    Full = toMono(Full)
    t = np.arange(len(Full[:, 0])) * 1.0 / Freq

    plt.figure('Figure')
    plt.plot(t, Full, 'r--')
    plt.plot(t, Part, 'b--')

    new = Full

    # new = np.absolute(Full - Part)
    for point in range(len(Full)):
        if (abs(Full[point] - Part[point]) > Part[point]):
            new[point] = Part[point]
        else:
            new[point] = (Full[point] - Part[point])

    # if (Full[point] < 0 & Part[point] < 0):
    #         new[point] = -abs(Full[point] - Part[point])
    #     elif (Full[point] > 0 & Part[point] < 0):
    #         new[point] = abs(Full[point] + Part[point])
    #     elif (Full[point] < 0 & Part[point] > 0):
    #         new[point] = -abs(Full[point] + Part[point])
    #     elif (Full[point] > 0 & Part[point] > 0):
    #         new[point] = abs(Full[point] - Part[point])
    #     else:
    #         new[point] = Full[point] - Part[point]
    plt.plot(t, new, 'g--')
    plt.show()
    return new


def compareAudio(Expected, Record):
    freq, record = wav.read(Record, True)
    freq, master = wav.read(Expected, True)

    record = toMono(record)
    master = toMono(master)

    record = record[-get_max_correlation(master, record):]

    if (len(record) > len(master)):
        a = len(record) - len(master)
        record = record[:-a]
    else:
        a = len(master) - len(record)
        master = master[:-a]

    t = np.arange(len(master[:])) * 1.0 / freq
    plt.figure('User vs Recording')
    xlabel('Time')
    ylabel('Amplitude')
    plt.plot(t, master, 'k--')
    plt.plot(t, record, 'b--')
    plt.savefig('comparedFig.png', format='png')

    plt.figure('User')
    xlabel('Time')
    ylabel('Amplitude')
    plt.plot(t, record, 'b--')
    plt.savefig('userFig.png', format='png')

    plt.figure('Master')
    xlabel('Time')
    ylabel('Amplitude')
    plt.plot(t, record, 'k--')
    plt.savefig('masterFig.png', format='png')

    totalError = 0
    for i in range(len(master)):
        totalError = totalError + abs(master[i] - record[i])


    errorCoeff = (totalError * freq)/ len(master)


    return int(errorCoeff)


# Voice = testGraph('Music Sample/Moves-Full.wav', 'Music Sample/Moves.wav')
# wav.write('Bruno.wav', 44100, Voice)

print(compareAudio('Music Sample/MovesK.wav', 'Music Sample/MovesA.wav'))
