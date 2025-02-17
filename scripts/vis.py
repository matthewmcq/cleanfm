import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sine wave
N = 441975  # FFT size
sample_rate = 48000
t = np.arange(N) / sample_rate
freq = 440  # A4 note
signal = np.sin(2*np.pi * freq * t)

df = pd.read_csv('../components.csv')
max_db = df['amplitude'].max()
df['amplitude_db'] = 20 * np.log10(df['amplitude'] / max_db + 1e-10)
df['decomp'] = 1


# Read the spectrum CSV
df_spectrum = pd.read_csv('../spectrum_before_ifft.csv')
df_spectrum['amplitude'] = df_spectrum['magnitude']
df_spectrum['phase'] = np.arctan2(df_spectrum['imag'], df_spectrum['real'])
# Normalize amplitudes using min-max normalization
max_val = df_spectrum['amplitude'].max()
min_val = df_spectrum['amplitude'].min()
df_spectrum['amplitude_normalized'] = (df_spectrum['amplitude'] - min_val) / (max_val - min_val)
df_spectrum['amplitude_db'] = 20 * np.log10(df_spectrum['amplitude_normalized'] / max_db + 1e-10)
df_spectrum['decomp'] = 3
df_spectrum['frequency_hz'] = df_spectrum['bin'] * sample_rate / len(df_spectrum)



# Compute FFT
y, sr = librosa.load('../examples/sine_wave_440hz.wav', sr=48000, mono=True)
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
window_size = 100
center_bin = int(freq * N / sample_rate) - 2000  # Convert 440Hz to bin number
bins = np.arange(center_bin-window_size, center_bin+window_size+1)
mask = (bins >= 0) & (bins < N//2)
bins = bins[mask]

ax.scatter(fft_data.real[bins], 
         fft_data.imag[bins], 
         bins,
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
          df_spectrum['bin'][bins],
          c='green', label='Spectrum', s=10, alpha=0.5)

# Generate and plot Dirichlet kernel
num_points = 10000
plot_bins = np.linspace(center_bin-window_size, center_bin+window_size, num_points)
kernel_points = np.zeros((len(plot_bins), 3))



for i, m in enumerate(plot_bins):
    diff = frequency - m
    if abs(diff) < 1e-16:
        value = np.exp(-1j * 0)  # Phase 0 for simple sine
    else:
        num = np.sin(np.pi * diff)
        denom = np.sin(np.pi * diff/N)
        value = (num/denom) * np.exp(-1j * (np.pi *diff) - 1j*phase)
    
    kernel_points[i] = [ amp* value.real, amp *value.imag, m]

ax.plot(kernel_points[:,0], 
       kernel_points[:,1], 
       kernel_points[:,2],
       'r-', label='Dirichlet Kernel', linewidth=1.5)

ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_zlabel('Frequency Bin')
ax.legend()

plt.title('3D Visualization of 440Hz Sine Wave FFT and Dirichlet Kernel')
plt.show()


# Read the CSV
df = pd.read_csv('../components.csv')
max_db = df['amplitude'].max()
df['amplitude_db'] = 20 * np.log10(df['amplitude'] / max_db + 1e-10)
df['decomp'] = 1

#remove lowest frequency


exit()




audio_path = '../examples/sine_wave_440hz.wav'
y, sr = librosa.load(audio_path, sr=48000, mono=False)

print(y.shape)
# left channel only
y = y[0]

fft = np.fft.fft(y)

len_original = len(fft)
# keep only first half of the fft
fft = fft
# normalize fft

magnitude = np.abs(fft)
#normalize magnitude
max_val = np.max(magnitude)
min_val = np.min(magnitude)
magnitude = (magnitude - min_val) / (max_val - min_val)
frequency = sr * np.linspace(0, 1, len_original)
phase = np.angle(fft)


# Create a DataFrame
df2 = pd.DataFrame({'frequency_hz': frequency,
                   'amplitude': magnitude,
                   'phase': phase,
                   'decomp': 0})

df2['amplitude_db'] = 20 * np.log10(df2['amplitude'] / max_db + 1e-10)

# join the two dataframes
df = pd.concat([df, df2], ignore_index=True)

audio_path = '../reconstructed_5k.wav'
y, sr = librosa.load(audio_path, sr=48000, mono=True)

fft = np.fft.fft(y)

len_original = len(fft)
# keep only first half of the fft
fft = fft
# normalize fft

magnitude = np.abs(fft)
#normalize magnitude
max_val = np.max(magnitude)
min_val = np.min(magnitude)
magnitude = (magnitude - min_val) / (max_val - min_val)
frequency = sr * np.linspace(0, 1, len_original)
phase = np.angle(fft)

# Create a DataFrame
df2 = pd.DataFrame({'frequency_hz': frequency,
                   'amplitude': magnitude,
                   'phase': phase,
                   'decomp': 2})

df2['amplitude_db'] = 20 * np.log10(df2['amplitude'] / max_db + 1e-10)

# join the two dataframes
df = pd.concat([df, df2], ignore_index=True)



# Add to your existing dataframe
df = pd.concat([df, df_spectrum], ignore_index=True)

# Update your plot to use different colors/markers for components vs spectrum
# plt.scatter(df[df['decomp'] == 3]['frequency_hz'], 
#            df[df['decomp'] == 3]['amplitude_db'],
#            s=5, alpha=0.4, color='red', label='Spectrum')
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

exit()

# Get top 5 frequencies by amplitude
top_5 = df.nlargest(5, 'amplitude')
print("Top 5 frequencies:")
for idx, row in top_5.iterrows():
    print(f"Frequency: {row['frequency_hz']:.2f} Hz, Amplitude: {row['amplitude_db']:.2f} dB")

# Create plot
plt.figure(figsize=(15, 8))

# Main scatter plot
scatter = plt.scatter(df['frequency_hz'], df['amplitude_db'], c=df['decomp'], 
                      cmap='viridis',
                     s=5, alpha=0.4, label='All components')

# Different colors for each fundamental and its harmonics
colors = ['r', 'g', 'b', 'c', 'm']
for i, (_, row) in enumerate(top_5.iterrows()):
    f0 = row['frequency_hz']
    # Plot the fundamental
    plt.scatter(f0, row['amplitude_db'], 
               color=colors[i], s=100, 
               label=f'F{i+1}: {f0:.1f} Hz')
    
    # Plot harmonics
    for n in range(2, 8):  # harmonics 2-7
        harmonic_freq = f0 * n
        plt.axvline(x=harmonic_freq, color=colors[i], 
                   linestyle='--', alpha=0.3)

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Frequency Spectrum with Harmonics of Top 5 Amplitudes')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()