import sys
from random import random as rand
import math

#### FUNCTIONS #########

###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
	assert len(u) == len(v), "dotproduct: u and v must be of same length"
	dp = 0
	for i in range(0, len(u), 1):
		dp += u[i]*v[i]
	return dp

###################
## Standardize the code here: divide each feature of each 
## datapoint by the length of each column in the training data
## return [traindata, testdata]
###################
def standardize_data(traindata, testdata):
    
    for i in range (rows):
        length =[0 for i in range(rows)]
        for j in range (cols):
            length[i] += traindata[i][j]**2
        length[i]=length[i]**0.5
        for j in range (cols):
            if length [i]!=0:
               ## print ('before standardize',traindata[i][j])
                traindata[i][j]=traindata[i][j]/length[i]
               ## print ('after standardize',traindata[i][j])

    for i in range (rowss):
        testlength =[0 for i in range(rowss)]
        for j in range (colss):
            testlength[i] += testdata[i][j]**2
        testlength[i]=testlength[i]**0.5
        for j in range (colss):
            testdata[i][j]=testdata[i][j]/testlength[i]

###################
## Solver for hinge loss
## return [w, w0]
###################
def hinge_loss(traindata, trainlabels):
    w=[]
    for j in range (cols):
        w.append(0.00002*rand()-0.0001)##initail the random w
    w0 = 0.00002*rand()-0.0001
    
    eta =0.001
    errors =[]
    idx = 0
    did =1
    while abs(did)>0.001:
        wcg = [0 for j in range(cols)]
        w0cg = 0
        for i in range (rows):
                dp = dotproduct(traindata[i], w) 
                if(trainlabels[i]*(dp+w0)<1):
                    for j in range(cols):
                        wcg[j] += -1 * trainlabels[i] * traindata[i][j]
                    w0cg += -1 * trainlabels[i]
        w0 = w0 - eta*w0cg
        for j in range (cols):
            w[j] = w[j] - eta*wcg[j]
            
        error = 0
        for  i in range (rows):
            error += max(0,1-trainlabels[i]*(dotproduct(w,traindata[i])+w0))
        errors.append(error)
        if idx == 0:
            idx+=1
            continue
        did = errors[idx]- errors [idx-1]
        idx +=1
        print('errors are ',did)
    return w,w0


#### MAIN #########
if __name__ == "__main__":
###################
#### Code to read train data and train labels
###################
    datafile =sys.argv[1]
    f = open(datafile)
    traindata = []
    trainlabels =[]
    i = 0
    l = f.readline()
    while (l!=''):
        a = l.split()
        l2 = []
        for j in range(len(a)):
            if(j==0):
                trainlabels.append(float(a[j]))
            else:
                l2.append(float (a[j]))
        traindata.append(l2)
        l2 =[]
        l = f.readline()
    
    cols = len(traindata[0])
    rows = len(trainlabels)
    f.close()

    lablefile = sys.argv[2]
    f = open(lablefile)
    testdata = []
    testlabels =[]
    i = 0
    l = f.readline()
    while (l!=''):
        b = l.split()
        l3 = []
        for j in range(len(b)):
            if(j==0):
                testlabels.append(float(b[j]))
            else:
                l3.append(float (b[j]))
        testdata.append(l3)
        l3 =[]
        l = f.readline()
    colss = len(testdata[0])
    rowss = len(testlabels)
    f.close()
standardize_data(traindata, testdata)

k = int(sys.argv[3])
test = []
train = []
for i in range(len(testdata)):
    p = []
    for j in range(k):
        p.append(0)
    test.append(p)

for i in range(len(traindata)):
    q = []
    for j in range(k):
        q.append(0)
    train.append(q)



for i in range(k):
    x = []
    y = []
    w = []
    for j in range(len(traindata[0])):
        w.append(random.uniform(-1, 1))

    for j in range(len(traindata)):
        x.append(dotproduct(traindata[j], w))


    w[0] = random.uniform(min(x), max(x))
    x = []
    for j in range(len(traindata)):
        x.append(dotproduct(traindata[j], w))
    for j in range(len(traindata)):
        if (x[j] > 0.0):
            train[j][i] = 1
        else:
            train[j][i] = 0
    for j in range(len(testdata)):
        y.append(dotproduct(testdata[j], w))
    for j in range(len(testdata)):

        if (y[j] > 0.0):
            test[j][i] = 1
        else:
            test[j][i] = 0

standardize_data(train, test)

[w_1,w0_1] = hinge_loss(traindata, trainlabels)

OUT = open("original_output.txt", 'w')
rows1 = len(testdata)
n1 = 0
for i in range(rows1):
    dp = dotproduct(w_1, testdata[i]);
    if (dp < 0):
        print("0", i, file = OUT);
        if testlabels[i] == -1:
            n1 = n1+1
    else:
        print("1",i, file = OUT);
        if testlabels[i] == 1:
            n1 = n1+1
print("Original Hinge Loss" + " the accuracy = " + str(round((n1 / rows1),2)))




[w,w0] = hinge_loss(train, trainlabels)

OUT = open("01space_output.txt", 'w')
rows = len(test)
n = 0
for i in range(rows):
    product = dotproduct(w, test[i]);
    if (product < 0):
        print("0", i, file = OUT);
        if testlabels[i] == -1:
            n = n+1
    else:
        print("1",i, file = OUT);
        if testlabels[i] == 1:
            n = n+1
print("Hinge Loss with k = " + str(k) + " the accuracy = " + str(round((n / rows),2)))


