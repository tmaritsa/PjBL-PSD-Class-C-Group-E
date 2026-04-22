import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import convolve, correlate

path = r"C:\Users\talit\OneDrive\Dokumen\PROGRAM\sound\main.wav"
audio, fs = sf.read(path)

if len(audio.shape) > 1:
    audio = audio[:, 0]

audio = audio / np.max(np.abs(audio))
t = np.arange(len(audio)) / fs

f = 300
det_signal = 0.5 * np.sin(2 * np.pi * f * t)

step = 1900
t_plot = t[::step]
det_plot = det_signal[::step]

add_signal = audio + det_signal
mul_signal = audio * det_signal
conv_signal = convolve(audio, det_signal, mode='same')
corr_signal = correlate(audio, det_signal, mode='same')

def normalize(x):
    return x / np.max(np.abs(x))

add_signal = normalize(add_signal)
mul_signal = normalize(mul_signal)
conv_signal = normalize(conv_signal)
corr_signal = normalize(corr_signal)

def rms(x):
    return np.sqrt(np.mean(x**2))

power = audio**2
clipping = np.abs(audio) >= 1

kernel = np.ones(500)/500
moving_avg = convolve(audio, kernel, mode='same')

print("RMS Audio:", rms(audio))
print("Clipping:", np.any(clipping))

plt.figure()
plt.plot(t, audio)
plt.title("Sinyal Audio")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t_plot, det_plot)
plt.title("Sinyal Deterministik")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, add_signal)
plt.title("Penjumlahan")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, mul_signal)
plt.title("Perkalian")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, conv_signal)
plt.title("Konvolusi")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, corr_signal)
plt.title("Korelasi")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, power)
plt.title("Daya Sinyal")
plt.xlabel("Waktu (s)")
plt.ylabel("Daya")
plt.grid()
plt.show()

plt.figure()
plt.plot(t, audio, label="Sinyal")
plt.plot(t[clipping], audio[clipping], 'r.', label="Clipping")
plt.title("Deteksi Clipping")
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo")
plt.legend()
plt.grid()
plt.show()

def fft_analysis(x):
    X = np.fft.fft(x)
    X = np.abs(X) / len(x)
    freq = np.fft.fftfreq(len(x), 1/fs)
    mask = freq >= 0
    return freq[mask], X[mask]

freq_audio, X_audio = fft_analysis(audio)
freq_det, X_det = fft_analysis(det_signal)

hamming = np.hamming(len(audio))
hann = np.hanning(len(audio))

audio_hamming = audio * hamming
audio_hann = audio * hann

freq_ham, X_ham = fft_analysis(audio_hamming)
freq_hann, X_hann = fft_analysis(audio_hann)

clipped = np.clip(audio, -0.3, 0.3)
freq_clip, X_clip = fft_analysis(clipped)

plt.figure()
plt.plot(freq_audio, X_audio)
plt.title("FFT Audio")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

plt.figure()
plt.plot(freq_det, X_det)
plt.title("FFT Deterministik")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

plt.figure()
plt.plot(freq_clip, X_clip)
plt.title("FFT Clipped")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

plt.figure()
plt.plot(freq_ham, X_ham)
plt.title("Window Hamming")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

plt.figure()
plt.plot(freq_hann, X_hann)
plt.title("Window Hanning")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

plt.figure()
plt.plot(freq_ham, X_ham, label="Hamming")
plt.plot(freq_hann, X_hann, label="Hanning")
plt.title("Perbandingan Window")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()
