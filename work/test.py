import math

def fn(dict_values):
    probs = {}
    z = sum([math.exp(x) for x in dict_values.values()])
    for index, _ in dict_values.items():
        probs[index] = math.exp(dict_values[index])/z 
    return probs

dv = {'a':0,'b':1}

print(fn(dv))






