import timeit
from numpy import random

def test(a,*args,b=2):
    print(a,b,args[0])


test(1,3)

