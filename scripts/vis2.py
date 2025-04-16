import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import soundfile as sf


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# Generate sine wave
N = 48000 # 48000 * 2# FFT size
sample_rate = 48000
t = np.arange(N) / sample_rate
freq = 200
modulator = 90
MOD_INDEX = 0.5
# signal = np.sin(2*np.pi * (1234) *t) + np.sin(2*np.pi * (1234 + modulator) *t)
# signal = np.sin(2*np.pi * (0) *t  + (np.sin(2*np.pi*(98)*t + 0.2725972 * np.sin(2*np.pi * (49) * t) + 0.2530317 * np.sin(2*np.pi * (25) * t)+ 0.2220396 * np.sin(2*np.pi * (6) * t)+ 0.2169014* np.sin(2*np.pi * (8) * t)+0.1802196* np.sin(2*np.pi * (13) * t)+ 0.1799041* np.sin(2*np.pi * (226) * t) +  0.17207* np.sin(2*np.pi * (171) * t) +  0.158741* np.sin(2*np.pi * (196) * t) + 0.686119* np.sin(2*np.pi * (1.0) * t ) + 0.396291* np.sin(2*np.pi * (2.0) * t ) + 0.3602* np.sin(2*np.pi * (3.0) * t )+ 0.238323* np.sin(2*np.pi * (4.0) * t ) + 0.1990* np.sin(2*np.pi * (5.0) * t )))) - np.sin(2*np.pi*98)
signal = np.sin(2*np.pi * (0) *t + 0.2725972 * np.sin(2*np.pi * (49) * t) + 0.2530317 * np.sin(2*np.pi * (25) * t)+ 0.2220396 * np.sin(2*np.pi * (6) * t) )#+ 0.2169014* np.sin(2*np.pi * (8) * t)+0.1802196* np.sin(2*np.pi * (13) * t) + 0.686119* np.sin(2*np.pi * (1.0) * t ) + 0.396291* np.sin(2*np.pi * (2.0) * t ) + 0.3602* np.sin(2*np.pi * (3.0) * t )+ 0.238323* np.sin(2*np.pi * (4.0) * t ) + 0.1990* np.sin(2*np.pi * (5.0) * t ))


def modulator(c):
    return np.sin(2*np.pi * (c) *t + 0.2725972 * np.sin(2*np.pi * (49) * t) +  0.2530317 * np.sin(2*np.pi * (25) * t)) 


# signal *= 0.2725972 * np.sin(2*np.pi * (49) * t) * 0.2530317 * np.sin(2*np.pi * (25) * t)* 0.2220396 * np.sin(2*np.pi * (6) * t)* 0.2169014* np.sin(2*np.pi * (8) * t)*0.1802196* np.sin(2*np.pi * (13) * t)* 0.1799041* np.sin(2*np.pi * (226) * t) *  0.17207* np.sin(2*np.pi * (171) * t) *  0.158741* np.sin(2*np.pi * (196) * t) * 0.686119* np.sin(2*np.pi * (1.0) * t ) * 0.396291* np.sin(2*np.pi * (2.0) * t ) * 0.3602* np.sin(2*np.pi * (3.0) * t )* 0.238323* np.sin(2*np.pi * (4.0) * t ) * 0.1990* np.sin(2*np.pi * (5.0) * t )

signal2 = np.sin(2*np.pi * (123.0) *t + 0.2725972 * np.sin(2*np.pi * (49) * t) + 0.2530317 * np.sin(2*np.pi * (25) * t)+ 0.2220396 * np.sin(2*np.pi * (6) * t)+ 0.2169014* np.sin(2*np.pi * (8) * t)+0.1802196* np.sin(2*np.pi * (13) * t)+ 0.1799041* np.sin(2*np.pi * (226) * t) +  0.17207* np.sin(2*np.pi * (171) * t) +  0.158741* np.sin(2*np.pi * (196) * t) + 0.686119* np.sin(2*np.pi * (1.0) * t ) + 0.396291* np.sin(2*np.pi * (2.0) * t ) + 0.3602* np.sin(2*np.pi * (3.0) * t )+ 0.238323* np.sin(2*np.pi * (4.0) * t ) + 0.1990* np.sin(2*np.pi * (5.0) * t ))
signal3 = np.sin(2*np.pi * (146.9) *t + 0.2725972 * np.sin(2*np.pi * (49) * t) + 0.2530317 * np.sin(2*np.pi * (25) * t)+ 0.2220396 * np.sin(2*np.pi * (6) * t)+ 0.2169014* np.sin(2*np.pi * (8) * t)+0.1802196* np.sin(2*np.pi * (13) * t)+ 0.1799041* np.sin(2*np.pi * (226) * t) +  0.17207* np.sin(2*np.pi * (171) * t) +  0.158741* np.sin(2*np.pi * (196) * t) + 0.686119* np.sin(2*np.pi * (1.0) * t ) + 0.396291* np.sin(2*np.pi * (2.0) * t ) + 0.3602* np.sin(2*np.pi * (3.0) * t )+ 0.238323* np.sin(2*np.pi * (4.0) * t ) + 0.1990* np.sin(2*np.pi * (5.0) * t ))
# signal += signal2 + signal3
# signal3 = 0 * t
# signal2 = 0*t
# signal = np.sin(2*np.pi * (97.5) *t + 0.2725972 * np.sin(2*np.pi * (49) * t) +  2*0.11169844 * np.sin(2*np.pi * (6) * t)+ 2*0.1090835* np.sin(2*np.pi * (8) * t)+2*0.09031387* np.sin(2*np.pi * (13) * t))
# signal = np.sin(2*np.pi * (1234) *t + 0.2725972 * (np.sin(2*np.pi * (49) * t) + 2*0.12751864 * np.sin(2*np.pi * (25) * t)))
# signal =  np.sin(2*np.pi * (1234) *t + 0.2725972 * np.sin(2*np.pi * (49) * t))
# signal = np.cos(2*np.pi * (103) *t ) + np.cos(2*np.pi * (206) *t ) + np.cos(2*np.pi * (309) *t ) + np.cos(2*np.pi * (412) *t ) + np.cos(2*np.pi * (515) *t ) + np.cos(2*np.pi * (618) *t ) + np.cos(2*np.pi * (721) *t ) + np.cos(2*np.pi * (824) *t ) + np.cos(2*np.pi * (927) *t ) + np.cos(2*np.pi * (1030) *t )
# signal = np.sin(2*np.pi *sample_rate/4 *t + MOD_INDEX * np.sin(2*np.pi * (modulator+1) * t) + (0) * np.sin(2*np.pi * (1.5) * t ) + (0) * np.sin(2*np.pi * (3) * t ) + 0 * np.sin(2*np.pi * 2.0*modulator * t)) 
# signal3 = np.cos(2*np.pi *freq *t + MOD_INDEX * np.sin(2*np.pi * modulator * t)) * np.cos(2*np.pi *freq *t + MOD_INDEX * np.sin(2*np.pi * 70 * t))
# signal_ = np.sin(2*np.pi * (0) *t + MOD_INDEX * np.sin(2*np.pi * modulator * t))
# signal2 = np.sin(2*np.pi * 10.5*57 *t +MOD_INDEX * np.sin(2*np.pi * 10.5*57 * t))

# signal = np.sign(np.sin(2*np.pi * freq * t))
y, sr = librosa.load('../examples/test_runtime/TEST_1s.wav', sr=48000, mono=False)
y=y[0]
# y = signal_

fft = np.fft.fft(signal)

copy_of_fft = fft.copy()


# first_half = fft[:N//2]
# second_half = np.flip(fft[N//2 -1 :N])
# plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), abs(first_half), label='fft1', color='red')
# plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), abs(second_half), label='fft1', color='blue')
# plt.show()
# second_half = np.flip(fft[N//2:])

# fft = fft[:N//2]*np.conj(np.flip(fft[N//2:]))
fft = fft / np.max(abs(fft))
fftmax = np.max(abs(fft))
print(np.max(abs(fft)))

fft2 = np.fft.fft(signal2)
fft2 = fft2 / np.max(abs(fft2))

fftmax2 = np.max(abs(fft2))
print(np.max(abs(fft2)))

fft3 = np.fft.fft(signal3)
fft3 = fft3 / np.max(abs(fft3))

fftmax3 = np.max(abs(fft3))
print(np.max(abs(fft3)))

ffty = np.fft.fft(y)
fftymax= np.max(abs(ffty))
ffty = ffty / np.max(abs(ffty))



fft_fft = np.fft.fft(abs(fft)[:N//2])
fft_ffty = np.fft.fft(abs(ffty)[:N//2])



# convolved_mags = abs(fft_fft) * abs(fft_ffty)
convolved_mags = fft_fft * fft_ffty
plt.plot(np.linspace(0, sample_rate // 2, sample_rate // 2 ), abs(convolved_mags), label='fft1', color='red')
plt.show()
convolved_mags = np.fft.ifft(convolved_mags)
plt.plot(np.linspace(0, sample_rate // 2, sample_rate  //2), abs(convolved_mags), label='fft1', color='blue')
plt.show()


# make conjugate symetric
convolved_mags = np.concatenate((convolved_mags, np.conj(np.flip(convolved_mags))))
plt.plot(np.linspace(0, sample_rate , sample_rate  ), convolved_mags, label='fft1', color='blue')
plt.show()
# keep phases from fft but use new mags
convolved = abs(convolved_mags) * np.exp(1j * np.angle(ffty))
convolvedfft = np.fft.fft(convolved)





ifft = np.fft.ifft(convolved)
convolved = np.real(ifft)
# convolved = convolved / np.max(abs(convolved))
ymax = np.max(abs(y))
#make convolved max same as ymax
convolved = convolved * (ymax / np.max(abs(convolved)))
convolved = convolved / 2*np.max(abs(convolved))
combined = y- convolved 
print(np.max(abs(convolved)))


print(np.max(abs(convolved)))

# write to wav
sf.write('convolved.wav', convolved, sample_rate)

audio,sr = librosa.load('convolved.wav', sr=48000, mono=False)
# subtract original from convolved
# combined = audio - y
sf.write('combined.wav', combined, sample_rate)



convoled_mags = abs(convolved)
convoled_mags = convoled_mags 



fftymax2 = abs(ffty)[np.argsort(abs(ffty)[:N//2])[-2]]
fftymax3 = abs(ffty)[np.argsort(abs(ffty)[:N//2])[-3]]

print(np.max(abs(fftymax)))
print(fftymax2)


freqs = np.linspace(0, sample_rate, sample_rate)

arr = []

for i, freq in enumerate(freqs):
    if i > 2000:
        break
    print(f"{i} / {len(freqs)}")
    sig = modulator(freq)
    sigfft = np.fft.fft(sig)
    sigfft = sigfft / np.max(abs(sigfft))
    sigfft[i] = 0
    sum_ = np.sum(abs(sigfft)[:N//2]*abs(ffty)[:N//2] / np.max(abs(ffty)[:N//2]))
    # sum_ -= np.sum(abs(sigfft)[:N//2] *abs(ffty)[:N//2])
    print(sum_)
    arr.append(sum_)
    
arr = np.array(arr)
arr = arr / np.max(arr)
# plt.plot(np.linspace(0, 2001, 2001), arr, label='fft1', color='red')
# plt.plot(np.linspace(0, sample_rate, 2001), abs(fft), label='fft1', color='green', linewidth = 0.5)
# plt.plot(np.linspace(0, sample_rate, 2001), abs(ffty), label='ffty1', color='blue', linewidth =0.5)
# plt.plot(np.linspace(0, sample_rate, 2001), abs(sigfft), label='fft1', color='purple', linewidth=2)
# plt.show()


toshow = np.zeros((sample_rate))
toshow[:len(arr)] = arr
toshow[-len(arr):] = np.flip(arr)
toshow = toshow / np.max(toshow)

mags = abs(np.fft.fft(y))
mags *= toshow
ifft2 = np.fft.ifft(mags * np.exp(1j * np.angle(ffty)))
ifft2 = np.real(ifft2)
ifft2 = ifft2 / np.max(abs(y))
combined2 = np.array(y) - np.array(ifft2)
# write to wav
sf.write('ifft2.wav', ifft2, sample_rate)
sf.write('combined2.wav', combined2, sample_rate)

plt.plot(np.linspace(0, sample_rate, sample_rate ), convolved, label='fft1', color='red', linewidth=0.7)
plt.plot(np.linspace(0, sample_rate, sample_rate ), y, label='fft1', color='blue')
plt.plot(np.linspace(0, sample_rate, sample_rate ), combined, label='fft1', color='green')
plt.plot(np.linspace(0, sample_rate, sample_rate ), ifft2, label='fft1', color='purple')
plt.plot(np.linspace(0, sample_rate, sample_rate ), combined2, label='fft1', color='orange')
plt.show()



        
        

# adjust = fftymax / fftmax
# print(adjust)
adjust = 1
adjust2 = fftymax2 / fftmax2 * 0.9
adjust3 = fftymax3 / fftmax3 * 0.9
print(adjust2)
plt.plot(np.linspace(0, sample_rate, sample_rate), abs(ffty)-adjust*abs(fft), label='fft1', color='red', linewidth=0.5)
# plt.plot(np.linspace(0, sample_rate, sample_rate), abs(ffty)*adjust*abs(fft) , label='fft1', color='purple', linewidth=2)
plt.plot(np.linspace(0, sample_rate, sample_rate), toshow , label='fft1', color='purple', linewidth=2)
plt.plot(np.linspace(0, sample_rate, sample_rate), convoled_mags , label='fft1', color='green', linewidth=2)
plt.plot(np.linspace(0, sample_rate, sample_rate), adjust2*abs(fft2), label='fft1', color='orange', linewidth = 0.5)
plt.plot(np.linspace(0, sample_rate, sample_rate), adjust*abs(fft), label='fft1', color='green', linewidth = 0.5)
# plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), adjust3*abs(fft3)[:N//2], label='fft1', color='yellow', linewidth = 0.5)
plt.plot(np.linspace(0, sample_rate, sample_rate), abs(ffty), label='ffty1', color='blue', linewidth =0.5)
plt.show()
# get first 100 peaks of fft



fft2 = np.fft.fft(y)
freq_bin = int(freq * N / sample_rate)


print(len(fft))
print(len(fft2))


fft_of_mags = np.fft.fft(abs(fft)[:N//2])
plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), fft_of_mags, label='fftom2', color='green')
plt.show()
fft_of_mags = np.fft.fft(abs(fft_of_mags)[:N//2])
plt.plot(np.linspace(0, sample_rate //2, sample_rate //2),fft_of_mags, label='fftom3', color='blue')
plt.show()
fft_of_mags = np.fft.fft(abs(fft_of_mags)[:N//2])
plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), fft_of_mags, label='fftom4', color='purple')
plt.show()
fft_of_mags = np.fft.fft(abs(fft_of_mags)[:N//2]) 
fft_of_mags /= np.max(abs(fft_of_mags))
plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), fft_of_mags, label='fftom5', color='orange')
plt.show()

# plt.plot(np.linspace(0, sample_rate //2, sample_rate //2), fft_of_mags, label='fft_of_mags', color='red')
# plt.show()
# get the first 100 peaks
def ispeak(i):
    if abs(fft_of_mags[i]) > abs(fft_of_mags[i-1]) and abs(fft_of_mags[i]) > abs(fft_of_mags[i+1]):
        return True
    return False

peaks = []
for i in range(1, len(fft_of_mags)//2):
    if ispeak(i):
        peaks.append(i)
# get the first 100 peaks

peaks = np.array(peaks)
peaks_all = peaks.copy()

peaks = peaks[np.argsort(abs(fft_of_mags)[peaks])[-100:]]
print(peaks)


# get the frequencies of the first 100 peaks
frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)[peaks]
print(frequencies)

top5 = np.argsort(abs(fft_of_mags)[peaks])[-10:]


max_peak = max(abs(fft_of_mags))
print(f"({MOD_INDEX}, {abs(fft_of_mags)[peaks][top5] / max_peak})")

total = np.sum(abs(fft_of_mags)[peaks][top5] / max_peak)
print(f"Total: {total}")
plt.scatter(peaks, abs(fft_of_mags)[peaks] / max_peak)
# plt.show()
exit()




# fft_of_mags = fft_of_mags / np.max(abs(fft_of_mags))

fft2_of_mags = np.fft.fft(abs(fft2)[:N//2])

# fft2_of_mags = np.fft.fft(abs(fft2_of_mags))

# plt.plot(np.linspace(0, sample_rate, sample_rate), np.real(fft2), label='fft2', color='green')
# fft2_of_mags = fft2_of_mags / np.max(abs(fft2_of_mags))
# plt.show()

# fft22 = np.fft.fft(abs(fft2))
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='green')
# plt.show()
fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='purple')
# plt.show()
fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='red')
# plt.show()
fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='blue')


fft2_4 = abs(fft2_of_mags)
# plt.plot(np.linspace(0, sample_rate//2, sample_rate//2), fft2_4, label='fft2_4', color='blue')
# plt.show()



# fft2_of_mags = fft2_of_mags / np.max(abs(fft2_of_mags))
# plt.show()

fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='purple')
plt.show()
fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='red')
plt.show()
# fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='blue')
# plt.show()
# fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='purple')
# plt.show()
# fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='red')
# plt.show()
# fft2_of_mags = np.fft.fft(abs(fft2_of_mags)[:N//2])
# plt.plot(np.linspace(0, len(fft2_of_mags), len(fft2_of_mags)), fft2_of_mags, label='fft2', color='blue')
# plt.show()

convoled = fft_of_mags * (fft2_of_mags)



ifft = np.fft.ifft(convoled)
# convoled = convoled / np.max(abs(convoled))


ifft = np.real(ifft)
# ifft = ifft / np.max(abs(ifft))
fft2 = fft2 / np.max(abs(fft2))
fft = fft / np.max(abs(fft))


plt.figure(figsize=(12, 6))
# plt.plot(fft_of_mags, label='fft_of_mags', color='purple')
# plt.plot(fft2_of_mags, label='fft2_of_mags', color='yellow')
plt.plot(np.linspace(0, sample_rate, sample_rate), ifft, label='ifft', color='blue')
plt.plot(np.linspace(0, sample_rate, sample_rate), fft, label='fft', color='red')
plt.plot(np.linspace(0, sample_rate, sample_rate), fft2, label='fft2', color='green')

# plt.plot(np.linspace(0, sample_rate, sample_rate), np.real(convoled), label='convolced', color='red')
plt.grid(True)
plt.legend()
plt.show()



mags = np.abs(fft)
fft_of_mags = np.fft.fft(mags)
# get max peak after the first peak


peaks = []
for i in range(1, len(fft_of_mags)//2):
    if abs(fft_of_mags[i]) > abs(fft_of_mags[i-1]) and abs(fft_of_mags[i]) > abs(fft_of_mags[i+1]):
        peaks.append(i)
        
# get the first peak
max_peak = max(peaks, key=lambda x: abs(fft_of_mags[x]))
fft_of_mags_peak = fft_of_mags[max_peak]
print(f"First peak: {max_peak} in hz = {max_peak * sample_rate / N}")
print(f"First peak value: {fft_of_mags[max_peak]}")
print(f"Second peak value: {fft_of_mags[max_peak]}")

# Compute the autocorrelation of mags
autocorr = np.correlate(mags, mags, mode='full')
autocorr = autocorr[autocorr.size // 2:]  # Keep only the second half

# Plot the autocorrelation
plt.figure(figsize=(12, 6))
plt.plot(autocorr, label='Autocorrelation', color='purple')
plt.title('Autocorrelation of Magnitude Spectrum')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.legend()
plt.show()



# Plot the magnitude spectrum
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
# in hz
plt.plot(np.linspace(0, sample_rate, sample_rate), mags, color='blue')

plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot the FFT of the magnitude spectrum
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, len(fft_of_mags), len(fft_of_mags)), np.abs(fft_of_mags), color='red')
plt.scatter(max_peak, fft_of_mags[max_peak], color='green', label='First Peak')
plt.title('FFT of Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()


# write the signal to a file
sf.write('fm_numpy.wav', signal, 48000)


y, sr = librosa.load('fm_numpy.wav', sr=48000, mono=False)


fft_data = np.fft.fft(y)
center_bin = int(freq * N / sample_rate)
print(center_bin)
window_size = 1000
bins = np.arange(center_bin-window_size, center_bin+window_size+1)
mask = (bins >= 0) & (bins < N//2)

# Generate and plot Dirichlet kernel
num_points = 100000
plot_bins = np.linspace(center_bin-window_size, center_bin+window_size, num_points)



fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(fft_data.real[bins], 
         fft_data.imag[bins], 
         bins * sample_rate / N,
         c='blue', label='FFT', s=10, alpha=0.5)

def generate_kernel(plot_bins, freq, N, amp, phase):
    kernel_points = np.zeros((len(plot_bins), 3))
    
    for i, m in enumerate(plot_bins):
        diff = freq - m
        if abs(diff) < 1e-16:
            value = np.exp(-1j * 0)  # Phase 0 for simple sine
        else:
            num = np.sin(np.pi * diff)
            denom = np.sin(np.pi * diff/N)
            value = (num/denom) * np.exp(-1j * (np.pi *diff) - 1j*(phase + np.pi * diff/N))
            
            mirror_diff = freq - (N - m)
            mirror_phase = phase + np.pi * mirror_diff/N
            num_mir = np.sin(np.pi * mirror_diff)
            denom_mir = np.sin(np.pi * mirror_diff/N)
            value_mirror = (num_mir/denom_mir) * np.exp(-1j * (np.pi *mirror_diff) - 1j*(mirror_phase))
            value_mirror = np.conjugate(value_mirror)
            
            
        
        kernel_points[i] = [ amp* value.real, amp *value.imag, m]
        kernel_points[i] += [ amp* value_mirror.real, amp *value_mirror.imag, 0]
        
    
    
    # for i in range(center_bin-window_size, center_bin+window_size):
    #     diff = freq - i
    #     if abs(diff) < 1e-16:
    #         value = np.exp(-1j * 0)  # Phase 0 for simple sine
    #     else:
    #         num = np.sin(np.pi * diff)
    #         denom = np.sin(np.pi * diff/N)
    #         real = (num/denom) * np.cos(np.pi * diff + phase )
    #         imag = (num/denom) * -1* np.sin(np.pi * diff + phase )
    #         # print(value)
    #     kernel[i] += [amp * real, amp * imag, 0]
    #     if i != 0 and kernel[i][-1] == 0:
    #         kernel[i] += [0, 0, i]
    
    return kernel_points



fm_componets_csv = 'fm_components.csv'

#get top 9 frequencies from the csv, already sorted


df_spectrum = pd.read_csv(fm_componets_csv)
# already sorted get top 9
frequencies_df = df_spectrum.nlargest(9, 'amplitude')
frequencies = frequencies_df['frequency_hz'].values
print(frequencies)

phases = [0, 0, 0]
kernel_points = np.zeros((len(plot_bins), 3))
for i, m in enumerate(plot_bins):
    kernel_points[i] = [0, 0, m]
    

for i, freq in enumerate(frequencies):
    bin_freq = int(freq * N / sample_rate)
    # find the amplitdue and phase in the df_spectrum for freq = frequency_hz
    print(df_spectrum[df_spectrum['frequency_hz'] == freq]['phase'].iloc[0])
    
    kernel = generate_kernel(plot_bins, bin_freq, N, df_spectrum[df_spectrum['frequency_hz'] == freq]['amplitude'].iloc[0], df_spectrum[df_spectrum['frequency_hz'] == freq]['phase'].iloc[0])
    kernel_points[:,0] += kernel[:,0]
    kernel_points[:,1] += kernel[:,1]
    # kernel_points[:,0] += kernel.real
    # kernel_points[:,1] += kernel.imag
    
ax.plot(kernel_points[:,0], 
       kernel_points[:,1], 
       kernel_points[:,2] * sample_rate / N,
       'r-', label=freq, linewidth=1.5)


print(f"N: {N}, sr: {sample_rate}")

ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_zlabel('Frequency Hz')
ax.legend()

plt.title('3D Visualization of 440Hz Sine Wave FFT and Dirichlet Kernel')
plt.show()

exit()


# Read the CSV
# df = pd.read_csv('../components.csv')
# max_db = df['amplitude'].max()
# df['amplitude_db'] = 20 * np.log10(df['amplitude'] / max_db + 1e-10)
# df['decomp'] = 1

#remove lowest frequency






# join the two dataframes
# make new df with just difference between their ampls at each bin




df = pd.concat([df, df2], ignore_index=True)

audio_path = '../test_dc.wav'
y, sr = librosa.load(audio_path, sr=48000, mono=True)

recon_audio = y.copy()

diff_audio= test_audio - recon_audio
# write the diff audio to a file
sf.write('../diff_audio.wav', diff_audio, 48000)
exit()


fft = np.fft.fft(y)

len_original = len(fft)
# keep only first half of the fft
fft = fft
# normalize fft

magnitude = np.abs(fft)
#normalize magnitude
max_val = np.max(magnitude)
min_val = np.min(magnitude)

frequency = sr * np.linspace(0, 1, len_original)
phase = np.angle(fft)

# Create a DataFrame
df3 = pd.DataFrame({'frequency_hz': frequency,
                   'amplitude': magnitude,
                   'phase': phase,
                   'decomp': 2})

df3['amplitude_db'] = 20 * np.log10(df3['amplitude'] / max_db + 1e-10)

# join the two dataframes
df = pd.concat([df, df3], ignore_index=True)





# Add to your existing dataframe
df = pd.concat([df, df_spectrum], ignore_index=True)

# Update your plot to use different colors/markers for components vs spectrum
plt.scatter(df[df['decomp'] == 3]['frequency_hz'], 
           df[df['decomp'] == 3]['amplitude_db'],
           s=5, alpha=0.4, color='red', label='Spectrum')
# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['frequency_hz'], df['phase'],
                      s=1,
                      c=df['decomp'], # Color by amplitude
                      cmap='viridis',
                      alpha=0.6)

plt.colorbar(scatter, label='Amplitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.xscale('log')
plt.title('Frequency vs Phase Relationship')

# Optional: Add grid
plt.grid(True, alpha=0.3)

plt.show()


# Get top 5 frequencies by amplitude
top_5 = df_spectrum.nlargest(5, 'amplitude')
print("Top 5 frequencies:")
for idx, row in top_5.iterrows():
    print(f"Frequency: {row['frequency_hz']:.2f} Hz, Amplitude: {row['amplitude_db']:.2f} dB")

# Create plot
plt.figure(figsize=(15, 8))

# normalize df1 to df2 
df1['amplitude'] = df1['amplitude'] / df1['amplitude'].max()
df2['amplitude'] = df2['amplitude'] / df2['amplitude'].max()

# sort df2 by frequency
df2 = df2.sort_values(by='frequency_hz').reset_index(drop=True)


# all df2 with amplitudes greater than their neighbors
# Identify peaks in df2
df2_peaks = df2[(df2['amplitude'] > df2['amplitude'].shift(1)) & 
                (df2['amplitude'] > df2['amplitude'].shift(-1))]

df2_peaks['decomp'] = 5

print("Peaks in df2:")
for idx, row in df2_peaks.iterrows():
    print(f"Frequency: {row['frequency_hz']:.2f} Hz, Amplitude: {row['amplitude_db']:.2f} dB")


# df1 = pd.concat([df1, df2], ignore_index=True)
# df1 = pd.concat([df1, df2_peaks], ignore_index=True)

# Main scatter plot
scatter = plt.scatter(df['frequency_hz'], df['amplitude_db'], c=df['decomp'], 
                      cmap='viridis',
                     s=4, alpha=0.4, label=df['decomp'], )
plt.colorbar(scatter, label='decomp')

# Different colors for each fundamental and its harmonics
# colors = ['r', 'g', 'b', 'c', 'm']
for i, (_, row) in enumerate(top_5.iterrows()):
    f0 = row['frequency_hz']
    # Plot the fundamental
    # plt.scatter(f0, row['amplitude_db'], 
    #            color=colors[i], s=100, 
    #            label=f'F{i+1}: {f0:.1f} Hz')
    
    # Plot harmonics
    # for n in range(2, 7):  # harmonics 2-7
    #     harmonic_freq = f0 * n
    #     plt.axvline(x=harmonic_freq, color=colors[i], 
    #                linestyle='--', alpha=0.6)
        
    # for n in range(2, 4):  # harmonics 2-7
    #     harmonic_freq = f0 / n
    #     plt.axvline(x=harmonic_freq, color=colors[i], 
    #                linestyle='--', alpha=0.2)

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Reconstructed Spectrum vs Original')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()