import numpy as np

import matplotlib.pyplot as plotter

beginTime = 0
endTime = 1.0
dTime = 0.01

p = 0.25


time = np.arange(beginTime, endTime, dTime)
amplitude = np.zeros(len(time))
for i in range(len(time)):
    if abs(time[i] - endTime / 2) <= p:
        amplitude[i] = 1

# Create subplot
figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)

# Time domain representation for sine wave 1
axis[0].set_title('Rectangle impulse')
axis[0].plot(time, amplitude)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

# Frequency domain representation
fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
tpCount = len(amplitude)
values = np.arange(int(tpCount / 2))
timePeriod = tpCount * dTime
frequencies = values / timePeriod

# Frequency domain representation
axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fourierTransform))
axis[1].set_xlabel('Frequency')
axis[1].set_ylabel('Amplitude')

plotter.show()
