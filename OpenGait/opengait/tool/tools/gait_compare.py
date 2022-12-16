import torch
import numpy as np

def getemb(data):
    return data["inference_feat"]

def computedistence(x, y):
    distance = torch.sqrt(torch.sum(torch.square(x - y)))
    return distance

def compareid(data,dict,threshold_value):
    embs = getemb(data)
    min = threshold_value
    id = None
    for key in dict:
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"],value)
                    if distance.float() < min:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        id = key
                        min = distance.float()
    if id is None:
        print("############## no id #####################")
    print("############## distance #####################")
    print(distance.float(), min)
    return id