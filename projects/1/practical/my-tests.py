import numpy as np

boxes = {(2, 1): 0, (3, 1): 1}
storage = {(2, 1): 0, (3, 1): 1, (10, 20): 2}
b = np.asarray(list(storage.keys()))
print(b)

print(b.shape)

for box_cordinate in boxes.keys():
    a = np.tile(box_cordinate, (len(storage), 1))
    print(a)
    f = a - b
    print(f)
    c = np.linalg.norm(f, axis=1)
    print(c)
    e = c.min()
    print(e)
