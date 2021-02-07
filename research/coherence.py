from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

fs = 1e2
N = 1e4
amp = 200
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
print(time)
print(len(time))
'''
b, a = signal.butter(2, 0.25, 'low')
x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
y = signal.lfilter(b, a, x)
x += amp*np.sin(2*np.pi*freq*time)
y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
'''
f0 = 10.0
f1 = 4.0
x = 3.0 * np.cos(2 * np.pi * f0 * time) + np.random.standard_normal(size=len(time))
y = 4.0 * np.cos(2 * np.pi * f1 * time) + np.cos(2 * np.pi * f0 * time) + 15 * np.random.standard_normal(size=len(time))

x = 3.0 * np.cos(2 * np.pi * f0 * time) + np.random.standard_normal(size=len(time))
y = np.cos(2 * np.pi * f1 * time) + np.cos(2 * np.pi * f0 * time)

'''
plt.plot(time, x)
plt.plot(time, y)
plt.xlabel('time')
plt.ylabel('input')
plt.show()
'''

f1, Pxy = signal.csd(y, y, fs, window="boxcar", nperseg=1000)
# f2, Pxx = signal.csd(x, x, fs, window="boxcar")
# f3, Pyy = signal.csd(y, y, fs, window="boxcar")
print(f1)
print(len(f1))
plt.plot(f1, np.abs(Pxy) / time[-1])
# plt.plot(f2, np.abs(Pxx))
# plt.plot(f3, np.abs(Pyy))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD [V**2/Hz]')
plt.show()
'''

f, Cxy = signal.coherence(x, y, fs, window="boxcar", nperseg=1000)
f1, Cxy1 = signal.coherence(x, y, fs, nperseg=512)
print(f)
print(len(f))
plt.plot(f, Cxy)
plt.plot(f1, Cxy1)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()

'''
