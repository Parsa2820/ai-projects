import time

s = time.time()

lst = []
for i in range(10000000):
    lst.append('x')

print(time.time() - s)
