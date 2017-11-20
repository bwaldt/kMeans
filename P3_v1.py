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


def randomSeeding(mat_full,numCenters):
    users = np.shape(mat_full)[0]
    print users
    samples = np.random.randint(users, size=numCenters)
    return mat_full[samples,:]




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
    centers = mat[centers,:]
    return centers
    

def kmeans(numCenters,mat_full,iterations,batchSize):

    metrics = np.zeros((6,iterations))
    for j in range(2):
	if j == 0:
	    centerPoints = seeding_kmeans_plusplus(numCenters,batchSize,mat_full)
	else:
	    centerPoints = randomSeeding(mat_full,numCenters)
        
	for i in range(iterations):
	    users = np.shape(mat_full)[0]
	    samples = np.random.randint(users, size=batchSize)
	    mat = mat_full[samples,:]
	    d = np.empty([np.shape(mat)[0], numCenters])
	    d.fill(99999)
		
	    # Calculate Distance of points to each Clusters
	    for idx in range(np.shape(centerPoints)[0]):
	        d[:,idx] =  np.linalg.norm(centerPoints[idx,:] - mat,axis=1)**2

	    clusterAssignments = np.argmin(d , axis =1) # Assign Points to clusters

	    d_l2 = np.empty([np.shape(mat)[0], numCenters])
	    d_l2.fill(99999)
		
	    for idx in range(np.shape(centerPoints)[0]):
	        d_l2[:,idx] = np.linalg.norm(centerPoints[idx,:] - mat,axis =1)


            for cluster in range(numCenters):
	        if cluster in clusterAssignments.tolist():
	            centerPoints[cluster,:] = np.mean(mat[np.where(clusterAssignments == cluster),:],axis=1) #update center

	    # Calculate Min/Max/Mean to new Clusters
	    #newD = np.empty([np.shape(mat)[0], len(centers)])                          
            #newD.fill(99999)
			
	    #for idx in range(np.shape(centerPoints)[0]):
		#    newD[:,idx] =  np.linalg.norm(centerPoints[idx,:] - mat,axis=1)**2
	    distances = np.min(d_l2,axis=1)
	    metrics[0+j,i] = np.mean(distances)
            metrics[2+j,i] = np.amin(distances)
            metrics[4+j,i] = np.amax(distances)

    plotMetrics(metrics,numCenters)
    np.save('data/metrics' + str(numCenters) + '.npy',metrics)

def plotMetrics(metrics,numCenters):
    length = np.shape(metrics)[1]
    x = np.linspace(0,length,length)

    plt.plot(x,metrics[0,:].T,'b--',label="Mean k-means++") #CONTAINS YOUR 3RD ROW
    
    plt.plot(x,metrics[1,:].T,'g--',label="Mean Random") #CONTAINS YOUR 3RD ROW
    plt.plot(x,metrics[2,:].T,'r--',label="Min k-means++") #CONTAINS YOUR 4TH ROW
    plt.plot(x,metrics[3,:].T,'c--',label="Min Random") #CONTAINS YOUR 5TH ROW
    plt.plot(x,metrics[4,:].T,'m--',label="Max k-means++") #CONTAINS YOUR 4T
    plt.plot(x,metrics[5,:].T,'y--',label="Max Random") #CONTAINS YOUR 5TH R    
    plt.legend(loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Distance")
    plt.savefig("metricsB" + str(numCenters) + ".jpg")
    plt.close()



if __name__ == "__main__":
  # readInFile()
    
    mat_full = np.load('copy/mat.npy')
    numOfCenters = [5,50,100,200,300,400,500]
    for k in numOfCenters:
	kmeans(k,mat_full,50,2500)


