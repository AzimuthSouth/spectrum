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
    # delimiter = corr_data.delimiter
    delimiter = ','
    flat_traces = []
    for line in corr_data.traces:
        for sgnl in line:
            flat_traces.append(sgnl)
    traces_set = set(flat_traces)
    all_traces_defined = set([sig in corr_data.add_traces.split(',') for sig in list(traces_set)])
    if all_traces_defined != {True}:
        print("Error! Some signals in array traces don't defined variable add_traces!")
        exit()

    # print(all_traces_defined)
    df = pipeline.read_data(filename, corr_data.sigs + ',' + corr_data.add_traces, delimiter)
    if type(df) == str:
        print(df)
        exit()

    print("Load data Ok")

    sigs = pipeline.get_signals(corr_data.sigs)
    add_traces = pipeline.get_signals(corr_data.add_traces)

    status, folders = pipeline.check_folders_tree(mode, sigs)
    print(status)

    lines = []
    try:
        f = open(flight + ".dat", 'r')
        lines = f.readlines()
        f.close()
    except IOError:
        print(f"Create file {flight}.dat")

    f = open(flight + ".dat", 'a')
    for folder in folders:
        if folder in lines:
            pass
        else:
            f.write(folder)
    f.close()
    os.chdir(os.getcwd() + '/' + mode)

    status = pipeline.processing_parameters_set(flight, df, sigs, corr_data.k, corr_data.hann, corr_data.eps,
                                                corr_data.traces, corr_data.dt_max,
                                                corr_data.class_min1, corr_data.class_max1,
                                                corr_data.class_min2, corr_data.class_max2, corr_data.m)
    print(status)
