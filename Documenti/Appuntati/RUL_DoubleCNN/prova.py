import numpy as np
import matplotlib.pyplot as plt

# Genera un segnale di esempio (puoi sostituire questo con i tuoi dati reali)
fs = 1000  # Frequenza di campionamento (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Creazione dell'asse del tempo
acceleration_signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(fs) * 0.5  # Segnale di esempio con rumore
print(acceleration_signal)
# Calcola la densità spettrale di potenza (PSD) usando la Trasformata di Fourier
freqs, psd = plt.psd(acceleration_signal, NFFT=fs, Fs=fs)

# Visualizza la PSD
plt.figure(figsize=(8, 4))
plt.plot(freqs, 10 * np.log10(psd))  # La PSD viene solitamente espressa in decibel (dB)
plt.xlabel('Frequenza (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid(True)
plt.title('Densità spettrale di potenza del rumore')
plt.show()
