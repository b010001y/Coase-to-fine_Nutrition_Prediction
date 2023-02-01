import sys
import json

input_file = "/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/dish_metadata_cafe1-gram1000_100.csv"
lines = ""
# 读取文件
with open(input_file, "r",encoding='utf-8') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
keys = lines[0].split(',')
line_num = 1
total_lines = len(lines)
# 数据存储
datas = []
while line_num < total_lines:
        values = lines[line_num].split(",")
        datas.append(dict(zip(keys, values)))
        line_num = line_num + 1
# 序列化时对中文默认使用的ascii编码.想输出真正的中文需要指定ensure_ascii=False
json_str = json.dumps(datas, ensure_ascii=False, indent=4)
# 去除\",\\N,\n 无关符号
result_data = json_str.replace(r'\"','').replace(r'\\N','').replace(r'\n','')
output_file = input_file.replace("csv", "json")
# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result_data)
    print("convert success")