import numpy as np

import matplotlib.pyplot as plotter

beginTime = 0
endTime = 10.0
dTime = 0.01

f0 = 3.0

time = np.arange(beginTime, endTime, dTime)
amplitude = 3 * np.cos(2 * np.pi * f0 * time)

# Create subplot
figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)

# Time domain representation for sine wave 1
axis[0].set_title('Sine wave with a frequency of 4 Hz')
axis[0].plot(time, amplitude)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

# Frequency domain representation
fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude) / 5))]  # Exclude sampling frequency
tpCount = len(amplitude)
values = np.arange(int(tpCount / 5))
timePeriod = tpCount * dTime
frequencies = values / timePeriod

# Frequency domain representation
axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, abs(fourierTransform))
axis[1].set_xlabel('Frequency')
axis[1].set_ylabel('Amplitude')

plotter.show()
