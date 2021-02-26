import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import prepare
import schematisation

df = pd.read_csv('data.txt')
col_names = df.columns
r, c = df.shape

dt = 0.02
time = np.arange(0, dt * r, dt)
dff = pd.DataFrame()
dff['time'] = time
dff['input'] = df['input']
dff.to_csv('input.txt', index=False)
smooth = prepare.smoothing_symm(dff, 'input', 3, 1)
remove_steps = schematisation.merge(smooth, 'input', 0)
all_extremes = schematisation.pick_extremes(remove_steps, 'input')
print(all_extremes.iloc[:20])
filt_extremes = schematisation.merge_extremes(all_extremes, 'input', 10)

# plt.plot(dff['time'], dff['input'], label='raw')
plt.plot(smooth['time'], smooth['input'], label='smooth')
plt.plot(remove_steps['time'], remove_steps['input'], label='-steps')
plt.plot(all_extremes['time'], all_extremes['input'], label='all_extremes')
plt.plot(filt_extremes['time'], filt_extremes['input'], label='filter_extremes')
plt.xlabel('time [s]')
plt.ylabel('input [-]')
plt.legend(loc='best')
plt.show()

'''
delta = np.mean(abs(time[:-1] - time[1:]))
s1 = df[col_names[0]].to_numpy()
s1_smooth = prepare.smoothing_symm(df, 'input', 100, 1)
s1_sm_cent = prepare.smoothing_delta_symm(df, 'input', 100, 1)
d = (max(s1) - min(s1)) * 0.005

# compare result_smoothing and r1
df1 = pd.read_csv('data_res.txt', delim_whitespace=True)
col_names1 = df1.columns
r1 = df1[col_names1[0]].to_numpy()[:682]
r2 = df1[col_names1[1]].to_numpy()

plt.plot(range(len(r1)), s1_smooth, '+', label='input')
plt.plot(range(len(r1)), r1, label='r1')
#plt.plot(time, s1, label='res')
'''


#plt.plot(time, r2, label='res')
#plt.plot(time, ds1, label='detrend')
# plt.plot(s1m[:, 0], s1m[:, 1], '+', label='merge')
# plt.plot(ext[:, 0], ext[:, 1], label='ext')
# plt.plot(time, s1, label='res')


# plt.plot(time, r1[:len(time)], label='r1')
#plt.plot(time, dr, label='sig-detrend')
#plt.plot(time, s1 - r2, label='sig-r2')
# plt.plot(time[:len(r1)], r1, label='r1')





#ff = scipy.fft.rfftfreq(len(s1), dt)



'''
ds1 = signal.detrend(s1, bp=[100, 150, 195, 390, 585])
#  bp=range(1, len(s1), int(len(s1) / 100))
dr = s1 - ds1
d = (max(s1) - min(s1)) * 0.005
print(d)
s1m = schematisation.merge(list(zip(time, s1)), 1, d)
ext = schematisation.pick_extremes(s1m, 1)



df = pd.DataFrame(list(zip(ff, abs(fs1))),
                      columns=['freq', 'fft'])
df.to_csv('data_res.txt', index=False)


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