import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


df = pd.read_csv('sample.txt')
time = df["Time"].to_numpy()
delta = abs(time[:-1] - time[1:])

col_names = df.columns
val1 = 1306.4

s2 = df[df["Time"] >= val1]

print(s2)
'''
s1 = df[col_names[1]].to_numpy()
f, g_xx = signal.csd(s1, s1, (1.0 / np.mean(delta)), window="hann", nperseg=len(s1))
plt.plot(f, np.abs(g_xx))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD [V**2/Hz]')
plt.show()

s2 = df[col_names[4]].to_numpy()
f, g_xx = signal.csd(s2, s2, (1.0 / np.mean(delta)), window="hann", nperseg=len(s1))
plt.plot(f, np.abs(g_xx))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD [V**2/Hz]')
plt.show()

f, g_xy = signal.csd(s1, s2, (1.0 / np.mean(delta)), window="hann", nperseg=len(s1))
plt.plot(f, np.abs(g_xy))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD [V**2/Hz]')
plt.show()

f, c_xy = signal.coherence(s1, s2, (1.0 / np.mean(delta)), window="hann", nperseg=len(s1))
plt.plot(f, np.abs(c_xy))
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()
'''