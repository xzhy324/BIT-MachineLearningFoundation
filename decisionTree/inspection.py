import sys
import os
import csv
from math import log2


def loadData(input_file):
    ret = []
    with open(input_file, 'r') as fp:
        tsvreader = csv.reader(fp, delimiter='\t')
        for line in tsvreader:
            ret.append(line)
    return ret


def saveData(output_file, entropy, error):
    with open(output_file, 'w') as fp:
        text = "entropy: {}\nerror: {}".format(entropy, error)
        fp.write(text)


def inspect(input_file, output_file):
    """
    数据集的第一行应为features和label的名称，最后一列为label
    """
    data = loadData(input_file)
    labels = [line[-1] for line in data[1:]]

    positive_label = labels[0]
    positive_num = 0
    total_num = len(labels)

    for label in labels:
        if label == positive_label:
            positive_num += 1

    p = positive_num / total_num
    entropy = -p * log2(p) - (1 - p) * log2(1 - p)
    error = 1 - p

    saveData(output_file,entropy,error)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("unmatched argument numbers")
        exit(0)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print("input file does not exist")
        exit(0)

    inspect(input_file, output_file)
