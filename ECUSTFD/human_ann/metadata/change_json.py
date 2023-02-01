import json

def read_ann_json(file):
    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']
    for i in range(0,len(annotations)):
        cal = annotations[i]['cal']
        cal = float(cal)/15
        annotations[i]['cal'] = cal
        fat = annotations[i]['fat/g']
        fat = float(fat)/2
        annotations[i]['fat/g'] = fat
        carb = annotations[i]['carb/g']
        carb = float(carb)/2
        annotations[i]['carb/g'] = carb
        pro = annotations[i]['protein/g']
        pro = float(pro)/2
        annotations[i]['protein/g'] = pro

    return annotations

def write_ann_json(file,obj):
    with open(file,'w') as f:
        json.dump(obj,f)

ann = read_ann_json('/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/rgb_test_gram1000_100.json')
write_ann_json('/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/rgb_test_gram1000_120.json',ann)

ann = read_ann_json('/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/rgb_train_gram1000_100.json')
write_ann_json('/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/rgb_train_gram1000_120.json',ann)