from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy


def hann_coefficient(r, n):
    return (1 - np.cos(2 * r * np.pi / n)) / 2


def correction_hann(data, par=None):
    res = []
    for i in range(len(data)):
        res.append(data[i] * hann_coefficient(i, len(data)))
    return res

def frequencies(n, dt):
    f = np.zeros(n)
    for i in range(n):
        f[i] = i / n / dt
    return f[:int(n / 2)]


fs = 1.02e2
N = 1e4
amp = 200
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
'''
b, a = signal.butter(2, 0.25, 'low')
x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
y = signal.lfilter(b, a, x)
x += amp*np.sin(2*np.pi*freq*time)
y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
'''
dt = 0.01
t = np.arange(0.0, 6.0, dt)

f0 = 10.0
f1 = 4.0
x1 = 3.0 * np.cos(2 * np.pi * f0 * t)
y1 = 4.0 * np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f0 * t)

x = 3.0 * np.cos(2 * np.pi * f0 * t) + np.random.standard_normal(size=len(t))
y = 4.0 * np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f0 * t) + 0.5 * np.random.standard_normal(size=len(t))
'''
df = pandas.DataFrame(list(zip(t, x1, y1, x, y)),
                      columns=['Time', 'clear10', 'clear4_10', 'noisy10', 'noisy4_10'])
df.to_csv('harmonic.txt', index=False)
'''
'''
plt.plot(time, x)
plt.plot(time, y)
plt.xlabel('time')
plt.ylabel('input')
plt.show()
'''

xx = correction_hann(x)
f2, Pxx2 = signal.csd(xx, xx, 1.0 / dt, window="boxcar", nperseg=len(t))

# f1, Pxy = signal.csd(x, x, 1.0 / dt, window="hann", nperseg=len(t))
# f2, Pxx = signal.csd(x, x, fs, window="boxcar")
# f3, Pyy = signal.csd(y, y, fs, window="boxcar")
# f = frequencies(len(x), dt)
# ff = scipy.fft.rfftfreq(len(x), dt)
# print(np.mean(ff - f))
# print((8.0 / 3) * abs(Pxx2) - abs(Pxy))
# plt.plot(f1, np.abs(Pxy))
# plt.plot(f2, (8.0 / 3) * np.abs(Pxx2))
# plt.plot(f2, np.abs(Pxx))
# plt.plot(f3, np.abs(Pyy))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('CSD [V**2/Hz]')
# plt.show()


f, Cxy = signal.coherence(x, y, (1.0 /dt), window="boxcar", nperseg=60)
f1, Cxy1 = signal.coherence(x, y, (1.0 / dt), nperseg=512)
ff = scipy.fft.rfftfreq(512, dt)
print("ff={}".format(ff))
print("f1-ff={}".format(np.mean(f1 - ff)))
print(len(f1))
print(len(ff))
# print(len(f1))
plt.plot(f, Cxy)
plt.plot(f1, Cxy1)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()


