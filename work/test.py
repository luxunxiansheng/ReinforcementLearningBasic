import timeit
from numpy import random


N = 5
d = 3
C = 5

W = random.rand(C,d)

result = random.rand(d,1)

wordvectors_list = [random.rand(d,1) for i in range(N)]
print(wordvectors_list)
start = timeit.timeit()
[W.dot(wordvectors_list[i]) for i in range(N)]
end = timeit.timeit()
print(end-start)

wordvectors_one_matrix = random.rand(d,N)
print(wordvectors_one_matrix)
start = timeit.timeit()
W.dot(wordvectors_one_matrix)
end = timeit.timeit()
print(end-start)

