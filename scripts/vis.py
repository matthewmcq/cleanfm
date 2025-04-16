import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import soundfile as sf


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# y, sr = librosa.load('../examples/TEST_INTRO_SHORT.wav', sr=48000, mono=True)

# #left channel

# resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

# # write

# sf.write('../resampled_sf.wav', resampled, 22050)
# exit()




# Generate sine wave
N = 48000 # 48000 * 2# FFT size
sample_rate = 48000
t = np.arange(N) / sample_rate
freq = 440  # A4 note
signal = np.sin(2*np.pi * freq * t)

df= pd.read_csv('../components.csv')
max_db = 1 #df['amplitude'].max() 
df['amplitude_db'] = 20 * np.log10(df['amplitude'] / max_db + 1e-10)
df['decomp'] = 1

df1 = df.copy()


audio_path = '../examples/test_runtime/TEST_2s.wav'

y, sr = librosa.load(audio_path, sr=48000, mono=False)
test_audio = y.copy()
N = len(y[0])
print(N)
print(y.shape)
# left channel only
y = y[0]

fft = np.fft.fft(y)

len_original = len(fft)
# keep only first half of the fft
fft = fft
# normalize fft

magnitude = np.abs(fft)
print("MAGNITUDE:xq ")
print(np.max(magnitude))
#normalize magnitude
max_val = np.max(magnitude)
min_val = np.min(magnitude)
min_val_og = min_val
max_val_og = max_val
# magnitude = (magnitude - min_val) / (max_val - min_val)
frequency = sr * np.linspace(0, 1, len_original)
phase = np.angle(fft)


# Create a DataFrame
df2 = pd.DataFrame({'frequency_hz': frequency,
                   'amplitude': magnitude,
                   'phase': phase,
                   'decomp': 0})

df2['amplitude_db'] = 20 * np.log10(df2['amplitude'] / max_db + 1e-10)


# Read the spectrum CSV
df_spectrum = pd.read_csv('../spectrum_before_ifft.csv')
df_spectrum['amplitude'] = df_spectrum['magnitude']
df_spectrum['phase'] = np.arctan2(df_spectrum['imag'], df_spectrum['real'])
# Normalize amplitudes using min-max normalization
max_val = df_spectrum['amplitude'].max()
print(max_val)
min_val = df_spectrum['amplitude'].min()
df_spectrum['amplitude_normalized'] = (df_spectrum['amplitude'])# - min_val) / (max_val - min_val)
df_spectrum['amplitude_db'] = 20 * np.log10(df_spectrum['amplitude_normalized'] / max_db + 1e-10)
df_spectrum['decomp'] = 3
df_spectrum['frequency_hz'] = df_spectrum['bin'] * sample_rate / len(df_spectrum)



# Compute FFT
y, sr = librosa.load('../examples/test_runtime/TEST_2s.wav', sr=48000, mono=True)
fft_data = np.fft.fft(y)

phase = df['phase'].values[0]
amp = df['amplitude'].values[0]
frequency = df['frequency_hz'].values[0] *  N / sample_rate
print(frequency)
print(phase)
# 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot actual FFT points around the frequency of interest
window_size = 10
center_bin = int(freq * N / sample_rate)  # Convert 440Hz to bin number
bins = np.arange(center_bin-window_size, center_bin+window_size+1)
mask = (bins >= 0) & (bins < N//2)
bins = bins[mask]

ax.scatter(fft_data.real[bins], 
         fft_data.imag[bins], 
         bins * sample_rate / N,
         c='blue', label='FFT', s=10, alpha=0.5)

# Plot spectrum data
spectrum_points = np.zeros((len(df_spectrum), 3))
for i, row in df_spectrum.iterrows():
    
    spectrum_points[i] = [row['real'], row['imag'], row['bin']]
    
spectrum_points = spectrum_points[center_bin-window_size:center_bin+window_size]
print(spectrum_points.shape)
print(bins.shape)


ax.scatter(df_spectrum['real'][bins],
         df_spectrum['imag'][bins],
          df_spectrum['bin'][bins]* sample_rate / N,
          c='green', label='Spectrum', s=10, alpha=0.5)

# Generate and plot Dirichlet kernel
num_points = 10000
plot_bins = np.linspace(center_bin-window_size, center_bin+window_size, num_points)
kernel_points = np.zeros((len(plot_bins), 3))

# print RMS of df_spectrum and df2
# normalize for comparison
#print total amplitude of df2
sum_df2 = df2['amplitude'].sum()
print("SUM DF2: ", sum_df2)

overall_max = max(df_spectrum['amplitude'].max(), df2['amplitude'].max())
# df_spectrum['amplitude'] /= overall_max


rmse = np.sqrt(np.mean((df_spectrum['amplitude'] - df2['amplitude'])**2))
print("RMSE: ", rmse)
sum_df2 = df2['amplitude'].sum()

df2_max = df2['amplitude'].max()
print("SUM DF2: ", sum_df2)
print(rmse / sum_df2)
# exit()


for i, m in enumerate(plot_bins):
    diff = frequency - m
    if abs(diff) < 1e-16:
        value = np.exp(-1j * 0)  # Phase 0 for simple sine
    else:
        num = np.sin(np.pi * diff)
        denom = np.sin(np.pi * diff/N)
        value = (num/denom) * np.exp(-1j * (np.pi *diff) - 1j*phase)
    
    kernel_points[i] = [ amp* value.real, amp *value.imag, m]


print(f"N: {N}, sr: {sample_rate}")
ax.plot(kernel_points[:,0], 
       kernel_points[:,1], 
       kernel_points[:,2] * sample_rate / N,
       'r-', label='Dirichlet Kernel', linewidth=1.5)

ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_zlabel('Frequency Hz')
ax.legend()

plt.title('3D Visualization of 440Hz Sine Wave FFT and Dirichlet Kernel')
plt.show()


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