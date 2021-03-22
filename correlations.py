from staff import pipeline
import os
import sys


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[1]))
    filename = os.path.basename(sys.argv[1])
    print(f"Processing file: {filename}")
    flight, mode = pipeline.parse_filename(filename)
    print(f"Current flight: {flight}, current mode: {mode}")
    import corr_data
    delimiter = ','
    df = pipeline.read_data(filename, corr_data.sigs + ',' + corr_data.add_traces, delimiter)
    if type(df) == str:
        print(df)
        exit()

    print("Load data Ok")

    sigs = pipeline.get_signals(corr_data.sigs)
    add_traces = pipeline.get_signals(corr_data.add_traces)

    status, folders = pipeline.check_folders_tree(mode, sigs)
    print(status)
    f = open(flight + ".dat", 'w')
    f.write(folders)
    os.chdir(os.getcwd() + '/' + mode)

    status = pipeline.processing_parameters_set(flight, df, sigs, corr_data.k, corr_data.hann, corr_data.eps,
                                                corr_data.traces, corr_data.dt_max,
                                                corr_data.class_min1, corr_data.class_max1,
                                                corr_data.class_min2, corr_data.class_max2, corr_data.m)
    print(status)
