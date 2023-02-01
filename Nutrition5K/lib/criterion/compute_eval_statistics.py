r"""Script to compute statistics on nutrition predictions.

This script takes in a csv of nutrition predictions and computes absolute and
percentage mean average error values comparable to the metrics used to eval
models in the Nutrition5k paper. The input csv file of nutrition predictions
should be in the form of:
dish_id, calories, mass, carbs, protein
And the groundtruth values will be pulled from the metadata csv file provided
in the Nutrition5k dataset release where the first 5 fields are also:
dish_id, calories, mass, carbs, protein

Example Usage:
"""
import csv
import json
import os
from os import path
import statistics
import sys
import time

DISH_ID_INDEX = 0

def ReadCsvData(filepath):
    # if not path.exists(filepath):
    #     raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 2):
            column = [row[i] for row in file]
            parsed_data[column[DISH_ID_INDEX]] = column
            f_in.seek(0)
    return parsed_data


# import  argparse
# def get_argv():
#     args = argparse.ArgumentParser()
#

def calMAE(pre_path,tmp):

    DATA_FIELDNAMES = ["dish_id", tmp]
    # if len(sys.argv) != 4:
    #   raise Exception("Invalid number of arguments\n\n%s" % __doc__)
    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    groundtruth_csv_path = './evaluation/data_gt/'+tmp+'_test_gt.csv'  # sys.argv[1]
    predictions_csv_path = pre_path  # sys.argv[2]

    groundtruth_data = ReadCsvData(groundtruth_csv_path)
    prediction_data = ReadCsvData(predictions_csv_path)

    groundtruth_values = {}
    err_values = {}
    output_stats = {}

    for field in DATA_FIELDNAMES[1:]:
        groundtruth_values[field] = []
        err_values[field] = []

    m = 0
    for dish_id in prediction_data['dish_id']:
        if dish_id == 'dish_id':
            continue
        m = m + 1
        k = groundtruth_data['dish_id'].index(dish_id)
        for i in range(1, len(DATA_FIELDNAMES)):
            groundtruth_values[DATA_FIELDNAMES[i]].append(float(groundtruth_data[DATA_FIELDNAMES[i]][k]))
            err_values[DATA_FIELDNAMES[i]].append(abs(
                float(prediction_data[DATA_FIELDNAMES[i]][m])
                - float(groundtruth_data[DATA_FIELDNAMES[i]][k])))
            # groundtruth_values[DATA_FIELDNAMES[i]].append(
            #     float(groundtruth_data[dish_id][i]))
            # err_values[DATA_FIELDNAMES[i]].append(abs(
            #     float(prediction_data[dish_id][i])
            #     - float(groundtruth_data[dish_id][i])))

    for field in DATA_FIELDNAMES[1:]:
        output_stats[field + "_MAE"] = statistics.mean(err_values[field])
        output_stats[field + "_MAE_%"] = (100 * statistics.mean(err_values[field]) /
                                          statistics.mean(groundtruth_values[field]))

    # with open(output_path, "w") as f_out:
    #     f_out.write(json.dumps(output_stats))

    return output_stats[tmp+"_MAE"],output_stats[tmp+"_MAE_%"]

if __name__ == '__main__':
    ele = ['cal','car','fat','mas','pro']
    all_path = os.listdir('/disk/btc010001/nutrition5k/data/file/')
    for path in all_path:
        if path[16:19] == 'pro':
            mass_MAE,mass_MAE_pre = calMAE(path,'protein')
            print(f'\t ' + path + f'_MAE:{mass_MAE:.3f}' + '/' + f'{mass_MAE_pre:.3f}' +'%' + '\n' )
