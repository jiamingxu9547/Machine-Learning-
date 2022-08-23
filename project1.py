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
## Solver for least squares (linear regression)
## return [w, w0]
###################
def least_squares(traindata, trainlabels):
    w=[]
    for j in range (cols):
        w.append(0.00002*rand()-0.0001)##initail the random w
    w0 = 0.00002*rand()-0.0001
   
    eta =0.001
    errors =[]
    idx = 0
    did =1
    while (abs(did)>0.001):
        wcg = [0 for j in range(cols)]
        w0cg = 0
        for i in range (rows):
                dp = dotproduct(traindata[i], w)
                w0cg += w0 - trainlabels[i] +dp
                for j in range(cols):
                        wcg[j] += (dp - trainlabels[i])*traindata[i][j]
        w0 = w0 - eta*w0cg
        for j in range (cols):
            w[j] = w[j] - eta*wcg[j]
            
        error = 0
        for  i in range (rows):
            error += (trainlabels[i]-dotproduct(w, traindata[i])-w0)**2
        errors.append(error)
        if idx == 0:
            idx+=1
            continue
        did = errors[idx] - errors [idx-1]
        idx +=1
        print('errors are ',did)
    return w,w0
###################
## Solver for regularized least squares (linear regression)
## return [w, w0]
###################
def least_squares_regularized(traindata, trainlabels):
    w=[]
    for j in range (cols):
        w.append(0.00002*rand()-0.0001)##initail the random w
    w0 = 0.00002*rand()-0.0001
   
    eta =0.001
    errors =[]
    idx = 0
    did =1
    lmd =0.01
    while abs(did)>0.001:
        wcg = [0 for j in range(cols)]
        w0cg = 0
        for i in range (rows):
                dp = dotproduct(traindata[i], w)
                w0cg += w0 - trainlabels[i] +dp
                for j in range(cols):
                        wcg[j] += (dp -trainlabels[i])*traindata[i][j]
        w0 = w0 - eta*w0cg-2*eta*lmd*w0
        for j in range (cols):
            w[j] = w[j] - eta * wcg[j]-2*lmd*eta*w[j]
        wlen=0
        for i in range(0, len(w), 1):
        	wlen += w[i]**2
        error = 0
        for  i in range (rows):
            error += (trainlabels[i]-dotproduct(w, traindata[i])-w0)**2 + lmd *wlen
        errors.append(error)
        
        if idx == 0:
            idx+=1
            continue
        did = errors[idx]- errors [idx-1]
        print('errors are ',did)
        idx +=1
    return w,w0

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

###################
## Solver for regularized hinge loss
## return [w, w0]
###################
def hinge_loss_regularized(traindata, trainlabels):
    w=[]
    for j in range (cols):
        w.append(0.00002*rand()-0.0001)##initail the random w
    w0 = 0.00002*rand()-0.0001

    eta =0.001
    errors =[]
    idx = 0
    did =1
    lmd =0.01
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
            w[j] = w[j]-eta*wcg[j]- 2 *eta* lmd * w[j]
        wlen=0
        for i in range(0, len(w), 1):
        	wlen += w[i]**2
                    
        error = 0
        for  i in range (rows):
            error += max(0,1-trainlabels[i]*(dotproduct(w,traindata[i])+w0)) + lmd * wlen
        errors.append(error)
        if idx == 0:
            idx+=1
            continue
        did = errors[idx] - errors[idx-1]
        idx +=1
        print('errors are ',did)
    return w,w0

###################
## Solver for logistic regression
## return [w, w0]
###################
def logistic_loss(traindata, trainlabels):
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
                w0cg += (1- 1/(1+math.exp(-1 * trainlabels[i]*(dp+w0))))*-1 * trainlabels[i]
                for j in range(cols):
                        wcg[j] +=(1- 1/(1+math.exp(-1 * trainlabels[i]*(dp+w0))))*-1 * trainlabels[i]* traindata[i][j]
                        
        w0 = w0 - eta*w0cg
        for j in range (cols):
            w[j] = w[j]-eta*wcg[j]
            
            
        error = 0
        for  i in range (rows):
            error += math.log(1+math.exp(-1 * trainlabels[i]*(dotproduct(w,traindata[i])+w0)))
        errors.append(error)
        if idx == 0:
            idx+=1
            continue
        did = errors[idx]- errors [idx-1]
        idx +=1
        print('errors are ',did)
    return w,w0


###################
## Solver for adaptive learning rate hinge loss
## return [w, w0]
###################
def hinge_loss_adaptive_learningrate(traindata, trainlabels):
    w=[]
    for j in range (cols):
        w.append(0.00002*rand()-0.0001)##initail the random w
    w0 = 0.00002*rand()-0.0001
    emp_risk = 0
    diff = 1
    eta=0.001
    while (diff > 0.001):
        prev=emp_risk
        wcg = [0 for j in range(cols)]
        w0cg = 0
        for i in range (rows):
                dp = dotproduct(traindata[i], w)
                if(trainlabels[i]*(dp+w0)<1):
                    for j in range(cols): 
                        wcg[j] += -1 * trainlabels[i] * traindata[i][j]
                    w0cg += -1 * trainlabels[i]

        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
        bestobj = 1000000000000

        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]

            # update w
            for j in range(0, cols, 1):
                w[j] -= eta * wcg[j]
            w0 = w0 - eta*w0cg

            # calculate error
            emp_risk = 0
            for i in range(0, rows):
            
                    emp_risk += max(0,1-trainlabels[i]*(dotproduct(w,traindata[i])+w0)) 

            obj = emp_risk
            if obj < bestobj:
                bestobj = obj
                best_eta = eta

            # update w
            for j in range(0, cols, 1):
                w[j] += eta * wcg[j]
            w0 = w0 + eta*w0cg

        if best_eta != None:
            eta = best_eta

        # update w
        for j in range(0, cols, 1):
                w[j] -= eta * wcg[j]
        w0 = w0 - eta * w0cg
            
        emp_risk = 0
        for i in range(0, rows):
          
                emp_risk += max(0,1-trainlabels[i]*(dotproduct(w,traindata[i])+w0)) 

        diff = abs(prev - emp_risk)
        print('errors are ',diff)
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
[w,w0] = least_squares(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT1 = open("least_squares_predictions", 'w')
for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0:
                print(" 1    ",format(i),file=OUT1)

            else:
                print("-1    ",format(i),file=OUT1)
                
[w,w0] = least_squares_regularized(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT2 = open("least_squares_regularized_predictions", 'w')
for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0:
                print(" 1    ",format(i),file=OUT2)

            else:
                print("-1    ",format(i),file=OUT2)

[w,w0] = hinge_loss(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT3 = open("hinge_loss_predictions", 'w')

for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0:
                print(" 1    ",format(i),file=OUT3)

            else:
                print("-1    ",format(i),file=OUT3)

[w,w0] = hinge_loss_regularized(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT4 = open("hinge_loss_regularized_predictions", 'w')
for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0:
                print(" 1    ",format(i),file=OUT4)

            else:
                print("-1    ",format(i),file=OUT4)

[w,w0] = logistic_loss(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT5 = open("logistic_loss_predictions", 'w')
for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0.5:
                print(" 1    ",format(i),file=OUT5)

            else:
                print("-1    ",format(i),file=OUT5)

[w,w0] = hinge_loss_adaptive_learningrate(traindata, trainlabels)

print(w) 
wlen = math.sqrt(w[0]**2 + w[1]**2)
dist_to_origin = abs(w[2])/wlen
print("Dist to origin=",dist_to_origin)

wlen=0.
for i in range(0, len(w), 1):
	wlen += w[i]**2
wlen=math.sqrt(wlen)
print("wlen=",wlen)	
OUT6 = open("hinge_loss_adaptive_learningrate_predictions", 'w')
for i in range(rowss):
        #if (class_[i] != None):
            dp = dotproduct(w, testdata[i])+w0
            if dp > 0:
                print(" 1    ",format(i),file=OUT6)

            else:
                print("-1    ",format(i),file=OUT6)

