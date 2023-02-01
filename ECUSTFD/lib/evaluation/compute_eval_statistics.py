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

def calMAE(pre_res,tmp):

    DATA_FIELDNAMES = ["id", tmp]
    # if len(sys.argv) != 4:
    #   raise Exception("Invalid number of arguments\n\n%s" % __doc__)
    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    groundtruth_csv_path = './evaluation/data_gt/'+tmp+'_test_gt.csv'  # sys.argv[1]
    #predictions_csv_path = pre_res  # sys.argv[2]
    #output_path = ticks + 'output_statistics.json'  # sys.argv[3]

    groundtruth_data = ReadCsvData(groundtruth_csv_path)
    #prediction_data = ReadCsvData(predictions_csv_path)

    groundtruth_values = {}

    output_stats = {}

    for field in DATA_FIELDNAMES[1:]:
        groundtruth_values[field] = []

    err_ME = []

    for pred in pre_res:
        food_name = pred[0]
        food_value = pred[1]
        if 'S' in food_name:
            food_n,temp = food_name.split('S')
        elif 'T' in food_name:
            food_n, temp = food_name.split('T')
        idx = groundtruth_data['id'].index(food_n)
        true_value = groundtruth_data[tmp][idx]
        ME = (float(food_value)-float(true_value))/float(true_value)
        err_ME.append([food_n,ME])

    err_values = {}
    all_err_values = {}
    for i in range(0, len(err_ME)):
        food_name = err_ME[i][0]
        true_name = ''.join([x for x in food_name if x.isalpha()])
        err_values[ food_name] = 100
        all_err_values[true_name] = [0,0]

    for i in range(0,len(err_ME)-1):
        if err_ME[i][0] == err_ME[i+1][0]:
            food_name = err_ME[i][0]
            min_value = min(abs(err_ME[i][1]),abs(err_ME[i+1][1]))
            if min_value == abs(err_ME[i][1]):
                idx = i
            else:
                idx = i + 1
            if (min_value<err_values[ food_name]):
                err_values[food_name] = min_value

    for food in err_ME:
        food_name = food[0]
        food_value = abs(food[1])
        real_food_name = ''.join([x for x in food_name if x.isalpha()])
        all_err_values[real_food_name][0] = all_err_values[real_food_name][0] + food_value
        all_err_values[real_food_name][1] = all_err_values[real_food_name][1] + 1

    final_err_values = {}
    for k,v in all_err_values.items():
        true_value = v[0]/v[1]
        final_err_values[k] = true_value*100

    return final_err_values

if __name__ == '__main__':
    mass_MAE,mass_MAE_pre = calMAE(path,'protein')

