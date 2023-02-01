import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

all_name =  ['cal/g','mass','fat/g','carb/g','protein/g']
# x = [141, 159, 166, 172, 177, 182, 188, 196, 203, 214,
#      143, 160, 167, 173, 177, 183, 189, 196, 203, 215,
#      144, 160, 168, 173, 178, 184, 189, 196, 205, 218,
#      149, 161, 168, 174, 178, 185, 189, 196, 206, 223,
#      150, 161, 168, 174, 178, 186, 190, 196, 207, 225,
#      152, 162, 170, 174, 179, 186, 190, 197, 208, 226,
#      153, 163, 171, 175, 179, 187, 191, 197, 209, 228,
#      153, 163, 171, 175, 179, 187, 192, 198, 210, 233,
#      154, 164, 172, 175, 180, 187, 194, 198, 210, 233,
#      155, 165, 172, 175, 180, 187, 194, 200, 211, 234,
#      156, 165, 172, 176, 181, 188, 195, 201, 211, 234,
#      158, 165, 172, 176, 182, 188, 195, 202, 213, 237]
#
# plt.hist(x, edgecolor='k', alpha=0.35) # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.show()

#for name in all_name:
##########total_cal############
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# print(data)
# arr = data.to_numpy()
# #data.dtype()
# print(arr)
# arr = arr[:,1];
# print(arr)
# index = [446,565,995,2213,2334,3568,4218]; #delete >2000 奇异值
# arr = np.delete(arr,index)
# arr.sort()
# print(arr)
# print(arr.size)
#
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# sig = np.arange(0,1,0.05)
# sig_res_cal = []
# for i in sig:
#     sig_res_cal.append(arr[math.ceil(arr.size*i)])
# print(sig_res_cal)
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####

# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins='auto', alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('cal/g')
# plt.ylabel('Frequency')
# plt.title('total_cal')
# plt.show()
##########total_cal############

# # ##########total_mass############
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# print(data)
# arr = data.to_numpy()
# #data.dtype()
# print(arr)
# arr = arr[:,2];
# print(arr)
# index = [479,508, 2213, 2538, 2548, 3568, 4218];        #delete >1000 奇异值
# print(arr[index])
# arr = np.delete(arr,index)
# arr.sort()
# print(arr)
# print(arr.size)
#
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# sig = np.arange(0,1,0.05)
# sig_res_mass = []
# for i in sig:
#     sig_res_mass.append(arr[math.ceil(arr.size*i)])
# print(sig_res_mass)
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# print(max(arr))
# #print(arr);
# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins=100, alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('mass')
# plt.ylabel('Frequency')
# plt.title('total_mass')
# plt.show()
#
# ##########total_mass############





# ##########total_fat############
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# print(data)
# arr = data.to_numpy()
# #data.dtype()
# print(arr)
# arr = arr[:,3];
# print(arr)
# index = [192,446,565,3568];        #delete >1000 奇异值
# print(arr[index])
# arr = np.delete(arr,index)
# arr.sort()
# print(arr)
# print(arr.size)
#
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# sig = np.arange(0,1,0.05)
# sig_res_fat = []
# for i in sig:
#     sig_res_fat.append(arr[math.ceil(arr.size*i)])
# print(sig_res_fat)
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# print(max(arr))
# #print(arr);
# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins='auto', alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('fat/g')
# plt.ylabel('Frequency')
# plt.title('total_fat')
# plt.show()
# # dish_1565033265	106.343002
# # dish_1551567573	875.541016
# # dish_1551567604	875.541016
# # dish_1551567508	853.218018
#
# ##########total_fat############


# ##########total_carb############
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# print(data)
# arr = data.to_numpy()
# #data.dtype()
# print(arr)
# arr = arr[:,4];
# print(arr)
# index = [446,479,508,565,995,1796,2213,2334,3568,4218,4331,4716];        #delete >1000 奇异值
# print(arr[index])
# arr = np.delete(arr,index)
# arr.sort()
# print(arr)
# print(arr.size)
#
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# sig = np.arange(0,1,0.05)
# sig_res_carb = []
# for i in sig:
#     sig_res_carb.append(arr[math.ceil(arr.size*i)])
# print(sig_res_carb)
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####

# print(max(arr))
# #print(arr);
# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins='auto', alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('carb/g')
# plt.ylabel('Frequency')
# plt.title('total_carb')
# plt.show()
#
# # dish_1551567573	506.078979
# # dish_1561739805	101.256134
# # dish_1551381990	129.636002
# # dish_1551567604	506.07901
# # dish_1551389588	732.300049
# # dish_1562012076	100.683945
# # dish_1560974769	844.568604
# # dish_1551389551	732.300049
# # dish_1551567508	502.362
# # dish_1551389458	717.660034
# # dish_1551382149	150.31601
# # dish_1551382179	150.31601
#
# ##########total_carb############


# ##########total_pro############
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# print(data)
# arr = data.to_numpy()
# #data.dtype()
# print(arr)
# arr = arr[:,5];
# print(arr)
# # index = [446,479,508,565,995,1796,2213,2334,3568,4218,4331,4716];        #delete >1000 奇异值
# # print(arr[index])
# # arr = np.delete(arr,index)
# arr.sort()
# print(arr)
# print(arr.size)
#
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####
# sig = np.arange(0,1,0.05)
# sig_res_pro = []
# for i in sig:
#     sig_res_pro.append(arr[math.ceil(arr.size*i)])
# print(sig_res_pro)
# #####每5%取一次分位数，分位数就是找到5%那个位置的数，对那个位置进行向上取整####

# print(max(arr))
# #print(arr);
# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins='auto', alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('pro/g')
# plt.ylabel('Frequency')
# plt.title('total_pro')
# plt.show()
#
#
# ##########total_pro############

# #########cafe2_total_mass############
# data= pd.read_csv('dish_metadata_cafe2.csv',error_bad_lines=False)
# arr = data.to_numpy()
# #data.dtype()
# print(arr.shape)
# arr = arr[:,2];
# print(arr)
# # index = [446,479,508,565,995,1796,2213,2334,3568,4218,4331,4716];        #delete >1000 奇异值
# # print(arr[index])
# # arr = np.delete(arr,index)
# print(max(arr))
# #print(arr);
# #plt.hist(arr, edgecolor='k', alpha=0.35)  # 设置直方边线颜色为黑色，不透明度为 0.35
# plt.hist(arr, bins='auto', alpha=0.5, histtype='stepfilled',
#          color='steelblue', edgecolor='none')
# plt.xlabel('cafe2_mass/g')
# plt.ylabel('Frequency')
# plt.title('total_cafe2_mass')
# plt.show()
#
# ##########total_mass############


# rgb_train_dir = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt";
# rgb_test_dir = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt"
# depth_train_dir = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits/depth_train_ids.txt"
# depth_test_dir = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits/depth_test_ids.txt"
# rgb_train_id ,rgb_test_id, depth_train_id,depth_test_id = [],[],[],[]
# id_path = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits";
# # train_io = open(train_dir)
# id_name = ["rgb_train_ids.txt","rgb_test_ids.txt","depth_train_ids.txt","depth_test_ids.txt"]
# for name in id_name:
#     with open(os.path.join(id_path,name)) as inputfile:
#         for line in inputfile:
#             if name[0] == 'r':
#                 if name[5] == 'r':
#                     rgb_train_id.append(line.strip().split(','))
#                 else: rgb_test_id.append(line.strip().split(','))
#             elif name[7] == 'r':
#                 depth_train_id.append(line.strip().split(','));
#             else: depth_test_id.append(line.strip().split(','))
#
#
#
# data= pd.read_csv('dish_metadata_cafe1 - 1.csv')
# arr = data.to_numpy()
# arr_id = arr[:,0];
# #arr_id = np.reshape(arr_id,[4768,1])
# #index = np.arange(0,4768)
# rgb_wait_train =[] ,rgb_wait_test=[],depth_wait_train = [],depth_wait_test = []
# rgb_train_id = np.array(rgb_train_id)
# rgb_test_id = np.array(rgb_test_id)
# depth_train_id = np.array(depth_train_id)
# depth_test_id = np.array(depth_test_id)
# for r_train_id,r_test_id,d_train_id,d_test_id in rgb_train_id,rgb_test_id,depth_train_id,depth_train_id:
#     r_train_pos = np.where(r_train_id == arr_id)[0]
#     #print(pos)
#     rgb_wait_train.append(arr[r_train_pos])


#rgb_train_id = np.resize(rgb_train_id,[4768])
#print(arr_id.shape,rgb_train_id.shape)
#print(index[rgb_train_id != arr_id])  #只会比较对应下标数组的元素，不可靠


#for i in rgb_test_id:

# print(rgb_train_id ,rgb_test_id, depth_train_id,depth_test_id)
# train_id =  train_io.readlines()
#print(train_id)


