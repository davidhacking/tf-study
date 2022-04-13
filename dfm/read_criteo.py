# encoding=utf-8
import csv

path = "/Users/winnieshi/Downloads/criteo_sampled_data.csv"


def feel_db():
    labels = []
    with open(path) as cf:
        rows = csv.reader(cf)
        for i, row in enumerate(rows):
            if i == 0:
                continue
            labels.append(row[0])
    total = len(labels)
    ones = 0
    zeros = 0
    for label in labels:
        if label == '0':
            zeros += 1
        elif label == '1':
            ones += 1
    print("total={}, ones={}, zeros={}".format(total, ones, zeros))


if __name__ == '__main__':
    feel_db()
