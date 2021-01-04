'''
# #############################################################
# https://www.cbcity.de/die-fft-mit-python-einfach-erklaert
# #############################################################

import datafile from LTSpice simulation
LTSpice setzt keine regelmässigen samplepunkte sondern optimiert diese pro Rechenleistung
die samplerate (dt bzw. fs) aus dem File ist daher nur gemittelt!

import datafile from Rigol
Rigol gibt die Samplefrequenz im Datenfile an
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# case = 'simu'

case = 'LT'
_file = 'boost.txt'  # _file = '3k+3M_50V.txt'

# case = 'Rigol'
# _file = 'csv.csv'

# 'hanning' 'hamming' 'blackman' 'kaiser'
# if kaiser+ -> 0=rectangle 5=hamming 6=hanning 8.6=blackmann
_window = 'hanning'
_kaiser_val = 0.3
window_size = 14
dc_removed = False
_filter = False

sft_factor = 1  # enhance spectrum

# #############################################################
# simulate Signal
if case == 'simu':
    t = np.linspace(0, 2 * np.pi, 50000, endpoint=True)
    f = 1000  # Frequency in Hz
    A = 50.0  # Amplitude in Unit
    sig = A * np.sin(2 * np.pi * f * t)  # Signal
    dt = t[1] - t[0]
# #############################################################

# #############################################################
# LT Spice import
# txt Data Format
# time	                    \t  V(n001)
# 0.000000000000000e+000    \t  1.353553e+002
# 1.043650696491705e-005    \t  1.376027e+002
# _data = np.loadtxt('dataLT6.txt', delimiter='\t', skiprows=1)
# _data = np.loadtxt('NewFile1.csv', delimiter=',', skiprows=2)
# _data *= 1.0
# t, sig = zip(*_data[:, :2])
if case == 'LT':
    # t, sig = np.loadtxt(_file, usecols=(0, 1), unpack=True, delimiter='\t', skiprows=1, max_rows=800000)
    t, sig = np.loadtxt(_file, usecols=(0, 1), unpack=True, delimiter='\t', skiprows=1)
    # dt = t[1] - t[0]
    # print('samples:', len(_data))
    dt = t[-1] / len(t)  # total length of sample sequence
# #############################################################

# #############################################################
# Rigol import
# csv Data Format
# 1 X,          CH2,    Start,          Increment,
# 2 Sequence,   Volt,   -1.000000e-03,  2.000000e-09
# 3 0.0,4.76e+00
# 4 1.0,4.80e+00
# start_V = starting Voltage
# dt = time increment
# t = timestamp
# sig = Signal Amplitude
if case == 'Rigol':
    start_V, dt = np.loadtxt(_file, usecols=(2, 3), unpack=True, delimiter=',', skiprows=1, max_rows=1)
    t, sig = np.loadtxt(_file, usecols=(0, 1), unpack=True, delimiter=',', skiprows=2, max_rows=50000)
    t *= dt
# #############################################################

# #############################################################
# remove DC from signal
if dc_removed is True:
    sig = sig - np.mean(sig)

print('T=%.15fs (overall Time)' % t[-1])
fs = 1.0 / dt  # scan frequency
print('dt=%.15fs (Sample step Time)' % dt)
print('fs=%.2fHz (Sample Frequency)' % fs)
# #############################################################

# #############################################################
# Filtern
# Aus der Sampling Rate fa ergibt sich unsere Nyquist Frequenz -> 1/2 * fa
# Aus einer vorhergegangenen FFT Analyse wissen wir, dass wir 50 Hz Stoersignal haben
# der maechtige Dreisatz sagt also 1/2*fa : 50 Hz --> 1 : x = _cutoff
# 48kHz -> 24kHz -> _cutoff =  50Hz / 24kHz = 0.00021
# sodass wir unseren Filterkern erstellen koennen
# (Stuetzstellen und Fenster sind willkuerlich gewaehlt - Verbesserungen gerne!)
if _filter is True:
    n = 31  # Anzahl Stuetzstellen
    kern = signal.firwin(n, cutoff=0.01, window="hamming")

    Smooth = signal.lfilter(kern, 0.01, sig)
# macht die ersten n Eintraege unbrauchbar... ggf. Constant / Zero padding der Eingangsdaten?
# #############################################################


# #############################################################
# fft with windowing through Hanning resp. Hamming resp. Blackman resp. Kaiser
# https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy.kaiser
#
if _window == 'hanning':
    windowing = np.hanning(len(sig))
if _window == 'hamming':
    windowing = np.hamming(len(sig))
if _window == 'blackman':
    windowing = np.blackman(len(sig))
if _window == 'kaiser':
    windowing = np.kaiser(len(sig), _kaiser_val)  # 0=rectangle 5=hamming 6=hanning 8.6=blackmann


# #############################################################

# #############################################################
# signal w/o windowing
if _window == '':
    Y = np.fft.fft(sig)
# #############################################################

# #############################################################
# signal w/ windowing
if _window != '':
    Y = np.fft.fft(windowing * sig)
# #############################################################

# #############################################################
# signal w/ windowing and filtering
if _filter is True:
    YSmooth = np.fft.fft(windowing * Smooth)
# #############################################################


N = int(len(Y) / 2)
print('Y =', len(Y))
print('Nyquist =', N)
X = np.linspace(0, fs / 2, N, endpoint=True)

# #############################################################
# time domain signal
plt.figure()
plt.subplot(311)
plt.plot(t, sig)
plt.title('Time Domain Signal', size='9', color='blue')
plt.ylim(np.min(sig) * 3, np.max(sig) * 3)
plt.xlabel('Time [$s$]', size='8')
plt.ylabel('Amplitude [$Unit$]', size='8')
plt.grid(axis='both', which='both', color='black', linestyle='-.', linewidth=0.1)
lowest_sig = min(sig)
largest_sig = max(sig)
plt.ylim(lowest_sig - 0.25 * lowest_sig, largest_sig + 0.25 * largest_sig)
plt.xlim(0, t[-1])
# #############################################################

# #############################################################
# frequency domain signal
plt.subplot(312)
plt.plot(X, 2.0 * np.abs(Y[:N]) / N)
if dc_removed is True:
    fft_title = 'Frequency Domain Signal, window: {} {}, DC removed'.format(_window, window_size)
else:
    fft_title = 'Frequency Domain Signal, window: {} {}'.format(_window, window_size)
plt.title(fft_title, size='9', color='blue')
plt.xlabel('Frequency [$Hz$]', size='8')
plt.ylabel('Amplitude [$Unit$]', size='8')

# supported values are 'linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
plt.yscale('symlog')
plt.xscale('log')
plt.grid(axis='both', which='both', color='black', linestyle='-.', linewidth=0.1)
# #############################################################


# #############################################################
# stft magnitude
# windowing is already done in np.fft -> Yhann Yhamm Yblack
plt.subplot(313)
amp = 2 * np.sqrt(2)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
f, t, Zxx = signal.stft(windowing * sig * sft_factor, fs, nperseg=window_size, noverlap=None)
largest_f = max(f)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='nearest') # 'gouraud', 'nearest', 'flat', 'auto'
#plt.ylim(0.0, 5000000)
sft_title = 'STFT Magnitude(factor {}), highest f = {:.2f}Hz'.format(sft_factor, largest_f)
plt.title(sft_title, size='9', color='blue')
plt.ylabel('Frequency [$Hz$]', size='8')
plt.xlabel('Time [$s$]', size='8')
plt.grid(axis='both', which='both', color='black', linestyle='-.', linewidth=0.1)
# #############################################################

'''
plt.annotate("FFT",
            xy=(0.0, 0.1), xycoords='axes fraction',
            xytext=(-0.8, 0.2), textcoords='axes fraction',
            size=30, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=0.2"))
'''

# plt.savefig('FFT.png',bbox_inches='tight', dpi=150, transparent=True)

# maximize window
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.subplots_adjust(hspace=0.5, wspace=0.1)
plt.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.95)
# plt.tight_layout()


'''
# #############################################################
# show blackmann etc. Info

hann = np.hanning(len(sig))
hamm = np.hamming(len(sig))
black = np.blackman(len(sig))
kaiser = np.kaiser(len(sig), 0)

plt.figure(figsize=(8, 3))
plt.subplot(141)
plt.plot(hann)
plt.title('Hanning')
plt.subplot(142)
plt.plot(hamm)
plt.title('Hamming')
plt.subplot(143)
plt.plot(black)
plt.title('Blackman')
plt.tight_layout()
plt.subplot(144)
#plt.plot(kaiser)

from numpy.fft import fftshift
window = np.kaiser(11, 0)
A = np.fft.fft(window, 2048) / 25.5
mag = np.abs(np.fft.fftshift(A))
freq = np.linspace(-2000, 50200, len(A))
response = 20 * np.log10(mag)
response = np.clip(response, -100, 100)
plt.plot(freq, response)
plt.title('Kaiser')

plt.tight_layout()

# #############################################################
'''

plt.show()
