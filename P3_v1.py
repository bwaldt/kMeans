import gzip
import pandas as pd
import numpy as np
import re
import random

def readInFile():
    a = np.empty([45000000, 6])
    print np.shape(a)
    f =  gzip.open('R6/ydata-fp-td-clicks-v1_0.20090501.gz', 'rb')
    i = 0
    for line in f:
        split1 = line.split('|')[1]
        split2 = split1.split('user')[1]
        split3 = re.sub("\d+:", " ", split2) 
        a[i,:] = np.fromstring(split3,sep=' ')
        i = i + 1 
    print np.shape(a)
    np.save('mat.npy', a)

def seeding_kmeans_plusplus(number):
    mat = np.load('copy/mat.npy')

#    mat = mat[1:1000,:]
    users = np.shape(mat)[0]
    centers = []
    c1 = random.randint(0,np.shape(mat)[0]-1)
    centers.append(c1)
 #   print min([np.linalg.norm(mat[x,]-mat[c,])**2 for c in [1,2]]) 
    for i in range(number):
	D2 = np.array([min([np.linalg.norm(mat[x,]-mat[c,])**2 for c in centers]) for x in range(users)])
	probs = D2 / D2.sum()
	cumSum = probs.cumsum()
	r = random.random()
	ind = np.where(cumSum >= r)[0][0]
	centers.append(ind)

    print centers

if __name__ == "__main__":
  # readInFile()
    seeding_kmeans_plusplus(100)
