import numpy as np 

a = np.arange(5,11).reshape(2,3)

print(a)

it = np.nditer(a,flags=['multi_index'])


for x in it:
    
    s = it.iterindex
    y = it.multi_index
    print(x)
    print(s)
    print(y)
        
