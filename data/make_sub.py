import numpy as np
import os

list = os.listdir('train')
list.sort()
print list
with open('sub.csv', 'w') as f:
    line = ','.join(list)
    f.write("image,%s\n" % line)
    a = np.load('out.npy')
    with open('fish_test.lst', 'r') as f_:
        lines = f_.readlines()
        for ind, line in enumerate(lines):
            image = line.split('\t')[-1].strip('\n')
            prob = ["%.10f" % x for x in a[ind,:]]
            prob = ",".join(prob)
            print ind, image, prob
            f.write("%s,%s\n" % (image, prob))



