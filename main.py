# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from staff import pipeline



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('harmonic.txt')
    pipeline.calc_set_of_signals_cross_spectrum(df, -1, "hann")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
