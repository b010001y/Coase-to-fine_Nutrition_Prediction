import json


def takeMass(elem):
    return elem['mass']

def takeCal(elem):
    return elem['calories']

def takeFat(elem):
    return elem['fat']

def takeCarb(elem):
    return elem['carb']

def takePro(elem):
    return elem['protein']

def get_anchor(segment,phase,t):
    if phase == 'train':
        with open('./dataset/nutrition_Bin_number' + str(segment) + '/change_' + t + '_train' + str(segment)+'.json', 'r') as f:
            json_data = json.load(f)
            ann = json_data['annotations']
            ls = list(ann)
    else:
        if phase == 'test':
            with open('./dataset/nutrition_Bin_number' + str(segment) + '/change_' + t + '_test' + str(segment)+'.json', 'r') as f:
                json_data = json.load(f)
                ann = json_data['annotations']
                ls = list(ann)

    if t == 'mass':
        ls.sort(key=takeMass)
    elif t=='calories':
        ls.sort(key=takeCal)
    elif t== 'fat':
        ls.sort(key=takeFat)
    elif t=='carb':
        ls.sort(key=takeCarb)
    elif t=='protein':
        ls.sort(key=takePro)

    picture_size = int(len(ls)/segment)

    idx = -1
    list_anchor = []

    while idx<len(ls):
        idx = idx+1
        min_ = ls[idx][t]
        idx = idx + picture_size-1
        if len(ls)-idx>=picture_size:
            max_ = ls[idx][t]
        else:
            max_ = ls[-1][t]
            idx = len(ls)
        anchor_set = (min_+max_)/2
        list_anchor.append((min_,max_,anchor_set))
        #print(min_,max_,anchor_set)

    i = -1
    j = 0
    for mass in ls:
        i = i+1
        if i >= picture_size:
            i = 0
            if j != segment-1:
                j = j+1
        section = [0] * segment
        section[j] = 1
        mass['section'] = section
        mass['offset'] = mass[t] - list_anchor[j][2]

    return list_anchor


