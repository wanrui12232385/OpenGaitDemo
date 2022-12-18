import torch
import numpy as np
from changejson import writejson, getid, getidlist

def getemb(data):
    return data["inference_feat"]

def computedistence(x, y):
    distance = torch.sqrt(torch.sum(torch.square(x - y)))
    return distance

def compareid(data,dict,jsonpath,threshold_value):
    embs = getemb(data)
    min = threshold_value
    id = None

    dic={}
    
    for key in dict:
        disnumb = 0
        selfmin = threshold_value
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"],value)
                    if distance.float() < min:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        id = key
                        min = distance.float()
                    if distance.float() < selfmin:
                        selfmin = distance.float()
                        disnumb = selfmin
        print(key)
        galleryid = getid(jsonpath, key)
        print(galleryid,key)
        stridlist = list(galleryid)
        temp = ''.join(stridlist)
        galleryidnum = int(temp)
        dic[galleryidnum] = disnumb
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    print("############## distance #####################")
    print(distance.float(), min)
    return id, dic_sort