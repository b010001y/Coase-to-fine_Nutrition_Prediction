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
        with open('../data/change_'+t+'_train.json', 'rb') as f:
            json_data = json.load(f)
            ann = json_data['annotations']
            ls = list(ann)
    else:
        if phase == 'test':
            with open('../data/change_'+t+'_test.json', 'rb') as f:
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

    return ls,picture_size,list_anchor

if __name__ == '__main__':
    train_ls,train_picture_number,train_anchor = get_anchor(10,'train','calories')
    test_ls, test_picture_number, test_anchor = get_anchor(10, 'test','calories')
    print(train_anchor)
    print(test_anchor)

