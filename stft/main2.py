import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt


def load_data():
    _data = []
    try:
        x = (np.genfromtxt('dataLT5.txt', dtype=None))
        N = len(x)
        print(N)

    finally:
        return x, N


x, N = load_data()

####################################################

t = np.linspace(0, 2*np.pi, 1000, endpoint=True)
f = 3.0 # Frequency in Hz
A = 100.0 # Amplitude in Unit
x = A * np.sin(2*np.pi*f*t) # Signal
N = t

####################################################

fs = 1 / (5e-3 / N)
print('fs: ', fs)

amp = 2 * np.sqrt(2)
print('amp: ', amp)
time = np.arange(N) / float(fs)
print('time: ', time)

ax = plt.subplot(3, 1, 1)
black = np.blackman(len(x))
plt.plot(time, black*x)

ax = plt.subplot(3, 1, 2)
# f, t, Zxx = signal.stft(x, fs, nperseg=1000)
# f, t, Zxx = signal.stft(x, fs, nperseg=128)
f, t, Zxx = signal.stft(black*x, fs, nperseg=N/4)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

ax = plt.subplot(3, 1, 3)
Y1 = np.fft.fft(x)
N1 = len(Y1)/2+1
plt.plot(2*np.abs(Y1[:int(N1)])/N)
plt.xlabel('Frequency ($Hz$)')

plt.show()
