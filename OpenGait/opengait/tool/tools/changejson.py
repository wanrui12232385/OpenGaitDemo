import json

def writejson(path, idlist, gallery=True):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    list = []
    for frame_id in idlist:
        tid = '{:03d}'.format(frame_id)
        list.append(tid)
    if gallery:
        jsondict['TEST_SET'] = list
    else:
        jsondict["COMPARE_SET"] = list
    with open(path, 'w') as write_f:
        json.dump(jsondict, write_f, indent=4, ensure_ascii=False)

def getid(path, id, gallery=True):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    return jsondict['TEST_SET'][id-1] if gallery else jsondict['COMPARE_SET'][id-1]

def getidlist(path, gallery):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    return jsondict['TEST_SET'] if gallery else jsondict['COMPARE_SET']
    