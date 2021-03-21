from staff import pipeline
import sys

if __name__ == '__main__':
    folders = open(sys.argv[1], 'r').read().splitlines()
    for fold in folders:
        print(pipeline.export_kip(fold))
    # print(pipeline.export_kip(sys.argv[1]))
