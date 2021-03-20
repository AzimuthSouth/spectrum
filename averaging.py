from staff import pipeline
import os
import sys
from os import walk


if __name__ == '__main__':
    print(sys.argv[1])
    curr_dir = os.path.dirname(sys.argv[1])
    os.chdir(curr_dir)
    _, _, filenames = next(walk(curr_dir))
    filenames = [file for file in filenames if '.txt' in file]
    print(f"Average files: {filenames}")
    df, status = pipeline.load_ave_files(filenames)
    print(status)

    # mean class to KIP calculation from 1 to m
    kip_class = 5
    dff, status = pipeline.calc_kip(df, kip_class)
    if dff is None:
        pass
    else:
        dff.to_csv('kip.dat', index=False, encoding='utf-8')
    print(status)




    '''
    os.chdir(os.path.dirname(sys.argv[1]))

    filename = os.path.basename(sys.argv[1])
    print(f"Processing file: {filename}")
    flight, mode = pipeline.parse_filename(filename)
    print(f"Current flight: {flight}, current mode: {mode}")
    '''