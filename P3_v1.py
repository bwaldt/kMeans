import gzip
import pandas as pd
import numpy as np
import re
import random
import progressbar
import matplotlib.pyplot as plt




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

def seeding_kmeans_plusplus(clusters, batchSize,mat_full):

    users = np.shape(mat_full)[0]
    print users
    samples = np.random.randint(users, size=batchSize)
    mat = mat_full[samples,:]
    centers = []
    d = np.empty([np.shape(mat)[0], clusters])
    d.fill(99999)
    idx = 0
    c = random.randint(0,np.shape(mat)[0]-1)
    centers.append(c)
    print c
    while len(centers) < clusters:    
        d[:,idx] =  np.linalg.norm(mat[c,] - mat,axis=1)**2
        idx += 1
        D2 = np.min(d,axis=1)
        probs = D2 / D2.sum()
        cumSum = probs.cumsum()
        r = random.random()
        c = np.where(cumSum >= r)[0][0]
        centers.append(c)
#	print len(centers)
    print centers
    return centers
    

def kmeans(centers,mat_full,iterations,batchSize):
    centerPoints = mat_full[centers,:]
    metrics = np.zeros((3,iterations))
    for i in range(iterations):
        users = np.shape(mat_full)[0]
	samples = np.random.randint(users, size=batchSize)
        mat = mat_full[samples,:]
	d = np.empty([np.shape(mat)[0], len(centers)])
	d.fill(99999)
	
	# Calculate Distance of points to each Clusters
	for idx in range(np.shape(centerPoints)[0]):
    	    d[:,idx] =  np.linalg.norm(centerPoints[idx,:] - mat,axis=1)**2

	clusterAssignments = np.argmin(d , axis =1) # Assign Points to clusters

	for cluster in range(len(centers)):
    	    centerPoints[cluster,:] = np.mean(mat[np.where(clusterAssignments == cluster),:],axis=1) #update center

	# Calculate Min/Max/Mean to new Clusters
	newD = np.empty([np.shape(mat)[0], len(centers)])                          
        newD.fill(99999)
        	
        for idx in range(np.shape(centerPoints)[0]):
            newD[:,idx] =  np.linalg.norm(centerPoints[idx,:] - mat,axis=1)**2
	distances = np.min(newD,axis=1)
	metrics[0,i] = np.mean(distances)
	metrics[1,i] = np.amin(distances)
	metrics[2,i] = np.amax(distances)

    plotMetrics(metrics)


def plotMetrics(metrics):
    length = np.shape(metrics)[1]
    x = np.linspace(0,length,length)

    plt.plot(x,metrics[0,:].T,'r--',label="Mean") #CONTAINS YOUR 3RD ROW
    plt.plot(x,metrics[1,:].T,'g--',label="Min") #CONTAINS YOUR 4TH ROW
    plt.plot(x,metrics[2,:].T,'b--',label="Max") #CONTAINS YOUR 5TH ROW
    plt.legend(loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.savefig("metrics.jpg")



if __name__ == "__main__":
  # readInFile()
    
    mat_full = np.load('copy/mat.npy')
    centers = seeding_kmeans_plusplus(4,300000,mat_full)
    kmeans(centers,mat_full,10,300000)


