from staff import pipeline
# "replace this strings with processing parameters and traces"
# 'MR' for mean/range, 'MM' for min/max
code = 'MM'

sigs = "В3"
add_traces = "тангаж,V_км/ч"


print(f"Processing signals: {pipeline.get_signals(sigs)}")
print(f"Tracing signals: {pipeline.get_signals(add_traces)}")

delimiter = ','

# prepare parameters for processing
# smoothing window sizes, length of array = number of signals
length = len(sigs.split(','))
k = [0] * length

# hann-weighting, length of array = number of signals
hann = [False] * length

# amplitude filter values, length of array = number of signals
eps = [0.0] * length

# array of additional tracing parameters, length of array = number of signals
# should be same for the picked mode for picked signal for all flights
traces = [['тангаж', 'V_км/ч']] * length

# array of time for averaging additional tracing parameters, length of array = number of signals
# None if unconditional averaging
dt_max = [None] * length

# array of global minimum for Mean & Range,  length of array = number of signals
class_min1 = [None] * length
class_min2 = [None] * length
# array of global maximum for Mean & Range, length of array = number of signals
class_max1 = [None] * length
class_max2 = [None] * length

# array of classes count, length of array = number of signals
m = [10] * length
