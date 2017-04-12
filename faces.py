import cPickle as pickle
import numpy as np
import scipy
from scipy import misc
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
# --------------------------
#part 2
from os import listdir
from os.path import isfile,join
import random
#---------------------------
# part 3
from scipy.misc import imread
#from rgb2gray import rgb2gray
import pandas as pd
from numpy import *
from numpy.linalg import norm
from numpy import matrix
from numpy import transpose
from numpy import shape
from numpy import array
from numpy import gradient
from numpy import divide
# ---------------------------
# part 4
import matplotlib
#import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#import matplotlib.cm as cm

# ---------------------------
from copy import copy, deepcopy
from scipy.misc import imsave

import urllib

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


#--------------------------------------------------------------------------------
# call get_data_actors.py to get images for male
# call get_data_actresses.py to get images for female


actresses = list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))
actors = list(set([a.split("\t")[0] for a in open("facescrub_actors.txt").readlines()]))
# total list of names
acts = actresses + actors

# list of all actors' name:
# ['Angie Harmon', 'Fran Drescher', 'Lorraine Bracco', 'Peri Gilpin', 'America Ferrera', 'Kristin Chenoweth', 'Gerard Butler', 'Michael Vartan', 'Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Daniel Radcliffe']


# get sets for one actor
# call get_sets # of names in acts array times


# -------------------------------------------------------------------
print "part 2"
 
# get_sets(100, 10, 10, "baldwin","resized")
def get_sets(training_number, validation_number, test_number, name, myfilepath):
    # creat/initialize sets
    training_set = []
    validation_set = []
    test_set = []
    a_act = []

    onlyfiles = listdir(myfilepath)
   # random.shuffle(onlyfiles)  no need to shuffle here, only shuffle one_act is better
    # filenames = baldwin76_cropped_grayscaled_resized.jpg
    for y in onlyfiles:
        if name in y:
            ind = onlyfiles.index(y)
            a_act.append(onlyfiles[ind]) #a_act store filenames for one name
#   random.shuffle(one_act) # whenever shuffle, get different number
#   training_set.extend(one_act[0:100])
#   validation_set.extend(one_act[100:110])
#   test_set.extend(one_act[110:120])

#---------other method with fixed random indexs
    np.random.seed(0)
    one_act = np.random.permutation(training_number+validation_number+test_number)
    
    training_ind = one_act[0:training_number]
    validation_ind = one_act[(training_number):(training_number+validation_number)]
    test_ind = one_act[(training_number+validation_number):(training_number+validation_number+test_number)]

    for x in training_ind:
        training_set.append(a_act[x])

    for y in validation_ind:
        validation_set.append(a_act[y])
    for z in test_ind:
        test_set.append(a_act[z])
    #print 'length of training set:',len(training_set)
    #print 'length of validation set:', len(validation_set)
    #print 'length of test set:', len(test_set)
 
    return training_set, validation_set, test_set

# ------------------------------------------------------------------------------
training_for_all = []
test_for_all = []
validation_for_all = []

for x in acts:
    name = x.split()[1].lower()
#    print name
    tra, vali, ts = get_sets(100,10,10,name,"resized")
    # use append instead of extend such that can access elements by index
    training_for_all.append(tra)
    test_for_all.append(ts)   
    validation_for_all.append(vali)
print(training_for_all)
print(test_for_all)
print(validation_for_all)

    
# 'Bill Hader', 'Steve Carell' called for part 3
# used random.shuffle so the data in the set is different each time called the function
hader_training, hader_validation,hader_test = get_sets(100, 10, 10, "hader","resized")
carell_training, carell_validation, carell_test = get_sets(100, 10, 10, "carell","resized")

#-------------------------------------------------------------------------------
#                                 PART 3
print "PART 3"
hc_training = hader_training + carell_training
hc_validation = hader_validation + carell_validation


def get_x_matrix(set):
    x_matrix = []
    for a in set:
        image = imread("resized/"+a,True)
        face = np.reshape(image,1024)
        for_theta0 = [1]
        for_theta0.extend(face)
        x_matrix.append(for_theta0)
    return array(x_matrix)
    
# get y matrix
# 0 for Bill Hader; 1 for Steve Carell
def get_y_matrix(row, column):
    y0 = np.zeros((row,column))
    y1 = np.ones((row,column))
    y_matrix = np.vstack((y0,y1))
    return array(y_matrix)
#print y_matrix
    
# guess theta
# TEST THE NUMBERS FOR THETA !!!!!!!!
def guess_theta(row,column):
    theta = np.ones((row,column))
    guess_num = 0.000005
    print " guess theta = ", guess_num
    theta = theta * guess_num
    return theta    

# cost function
def f(x_matrix,y_matrix,theta):
    m = len(x_matrix) # number of rows    
    # n = len(x_matrix[0]) #number of columns
    # print m
    # cost = 1/(2*m) * sum((y_matrix - dot(theta,array(x_matrix).T))**2)
    print "x_matrix shape", np.shape(x_matrix)
    cost = sum((y_matrix.T - dot(theta,array(x_matrix).T))**2)
    return cost
    
def df(x_matrix, y_matrix, theta):
    m = len(x_matrix)
   # d_cost= -2*(1/(2*m))* sum((y_matrix - dot(theta.T,x_matrix))*x_matrix,1)
    d_cost= -2* sum((y_matrix.T - dot(theta, array(x_matrix).T))*x_matrix.T,1)
    # print d_cost
    return d_cost
    


# CALL FUNCTION -> get x_matrix  
x_matrix_training = get_x_matrix(hc_training)
x_matrix_validation = get_x_matrix(hc_validation)

y_matrix_training = get_y_matrix(100,1)     #output y_matrix shape (200,1)
y_matrix_validation  = get_y_matrix(10,1)

theta_training =   guess_theta(1, 1025)                 
# print type(theta_training)


# CALL COST FUNC f FOR TRAINING_SET & VALIDATION_SETas
# be careful with dimensions

cost_training = f(x_matrix_training, y_matrix_training,theta_training)
print "training set cost:", cost_training
print "\n"
cost_validation = f(x_matrix_validation,y_matrix_validation,theta_training)
print "validation set cost:", cost_validation


gd_training = df(x_matrix_training,y_matrix_training,theta_training)
# print "gd_training:", gd_training
print "\n"
# DONT DO GD ON VALIDATION SET, USE THETA GOT FROM TRAINING SET

print "find gd"
def grad_descent(f,df,x_matrix,y_matrix,init_theta,alpha):
    EPS = 1e-8 #EPS = 10*(-5) machine epsilon
    prev_theta = init_theta-10*EPS
    theta = init_theta.copy()
    max_iter = 30000
    iter = 0
    while norm(theta - prev_theta) > EPS and iter < max_iter:
        prev_theta = theta.copy()
        theta -= alpha*df(x_matrix,y_matrix,theta)
        if iter % 1500 == 0:
#            print "\n"
            print "Iteration", iter
#            print "Gradient: ", df(x_matrix,y_matrix, theta), "\n"
        iter += 1
#    print "theta is", theta
    print "theta size", np.shape(theta)
    print "iter = ", iter
    return theta

# ..........................
alpha = 0.00000000001
print "choose alpha = ",alpha

theta0_training = np.zeros((1,1025))                                           # MARK


theta_training = grad_descent(f,df,x_matrix_training,y_matrix_training,theta0_training,alpha)
print "theta for training set",theta_training
print "\n"

# ............................
 
def predict_y(x_matrix, y_matrix, theta):
    hypothese = (dot(theta,x_matrix.T))
    y_matrix.T
    # print np.shape(y_matrix)
    # print np.shape(hypothese)
    # uncomment to see/print hypothese matrix
    # print hypothese # n by 1025 matrix; n = sample size; each row stands for one image
    pred_y_matrix = []
    for i in range(0,len(y_matrix)):
        if hypothese [0][i] <= 0.5:
            pred_y_matrix.append([0])
# METHOD FOR FINDING OUTPUT OF EITHER BILL HADER OR STEVE CARELL  ->see print           
            print "predicted: Bill Hader"
        else:
            pred_y_matrix.append([1])
            print "predicted: Steve Carell"            
    print pred_y_matrix
    print len(pred_y_matrix)
    return pred_y_matrix
    
def get_accuracy(pred_y_matrix, y_matrix):    
    right_guess = 0
    for i in range(0,len(pred_y_matrix)):
        if pred_y_matrix[i] == y_matrix[i]:
            right_guess += 1
    print "right guess times:", right_guess

    acc = float(right_guess)/(len(pred_y_matrix))
    return acc

print "\n"
pred_y_training = predict_y(x_matrix_training,y_matrix_training, theta_training)
accuracy_training = get_accuracy(pred_y_training,y_matrix_training)
print "accuracy training:", accuracy_training
print "\n"

# use theta_training got from gd (training set)
pred_y_validation = predict_y(x_matrix_validation,y_matrix_validation, theta_training)
accuracy_validation = get_accuracy(pred_y_validation,y_matrix_validation)
print "accuracy validation:", accuracy_validation
print "\n"

# ------------------------------------------------------------------------------
#                               PART 4

# THETA RESHAPE

print "theta_training from part 3:", theta_training
print np.shape(theta_training)

theta_image = []
for i in range(1,len(theta_training[0])):
    theta_image.append(theta_training[0][i])
# print array(theta_image)

plot_theta200  = reshape(theta_image,(32,32))
print np.shape(plot_theta200)
print plot_theta200


# display thetas in an image:
plt.figure(1)
plt.subplot(211)
#ax = fig.add_subplot(2,1,1)
plt.imshow(plot_theta200,cmap=cm.coolwarm)
# plt.colorbar()

# ------------------------------------------------------------------------------
 
# Only two images for training dataset

# CALL FUNCSTIONS FOR TWO IMAGES DATABASE CASE

arr2 = [hader_training[0],carell_training[0]]
two_x = get_x_matrix(arr2)

two_y = get_y_matrix(1,1)

two_theta = guess_theta(1,1025)
two_cost = f(two_x,two_y,two_theta)
two_d_cost = df(two_x,two_y,two_theta)

# two_ini_theta = np.zeros(2,2)

two_gd_theta = grad_descent(f,df,two_x,two_y,two_theta,alpha)
# plot the theta (two images database)
theta2_image = []
for i in range(1,len(two_gd_theta[0])):
    theta2_image.append(two_gd_theta[0][i])
# print array(theta_image)

plot_theta2  = reshape(theta2_image,(32,32))
print np.shape(plot_theta2)
print plot_theta2


# display thetas in an image:
plt.subplot(212)
#ax = fig.add_subplot(2,1,2)

plt.imshow(plot_theta2,cmap=cm.coolwarm)
# plt.colorbar()
plt.savefig('part4_theta_training_set.jpg')

# ------------------------------------------------------------------------------
#                           PART 5

act6 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def get_diff_set(training_set_size, validation_set_size, test_set_size,act6):
    training_set = []
    validation_set = []
    test_set = []
    for x in act6:
        names = x.split()[1].lower()
        print names
        tra,vali,test = get_sets(training_set_size, validation_set_size, test_set_size, names,"resized")
        training_set.extend(tra)
        validation_set.extend(vali)
        test_set.extend(test)
    return training_set, validation_set, test_set

# zero for female; one for male
# 0, 0, 0, 1, 1, 1

def get_diff_y(row,coln):
    ym6 = get_y_matrix(row*3,coln)
    return ym6


# CALL COST FUNC f FOR TRAINING_SET & VALIDATION_SETas
# be careful with dimensions

def predict_gender(x_matrix, y_matrix, theta):
    hypothese = (dot(theta,x_matrix.T))
    y_matrix.T
    # print np.shape(y_matrix)
    # print np.shape(hypothese)
    # uncomment to see/print hypothese matrix
    # print hypothese # n by 1025 matrix; n = sample size; each row stands for one image
    pred_y_matrix = []
    for i in range(0,len(y_matrix)):
        if hypothese [0][i] <= 0.5:
            pred_y_matrix.append([0])
# METHOD FOR FINDING OUTPUT OF EITHER BILL HADER OR STEVE CARELL  ->see print           
            print "predicted: female"
        else:
            pred_y_matrix.append([1])
            print "predicted: male"            
    print pred_y_matrix
    print len(pred_y_matrix)
    return pred_y_matrix

# ******************************************************************************    
# RUN FOR act6
size = []
accu_tr = []
accu_vali = []
cost_training_sizes = []
cost_validation_sizes = []

theta_training_sizes = []
        
for i in range (1, 21):
    num = 5*i
    print num
    size.append(num)
    myfilepath = "resized"    
    act6_tr, act6_vali, act6_ts = get_diff_set(num, 10, 10, act6)
        
    xm6_tr = get_x_matrix(act6_tr)
    xm6_vali = get_x_matrix(act6_vali)
        
    ym6_tr = get_diff_y(num,1)
    ym6_vali = get_diff_y(10,1)
    
    theta_training =   guess_theta(1, 1025)
        
    cost_training = f(xm6_tr, ym6_tr,theta_training)
    cost_training_sizes.append(cost_training)
        
    #    print "training set cost:", cost_training
    cost_validation = f(xm6_vali,ym6_vali,theta_training)
    cost_validation_sizes.append(cost_validation)
    #    print "validation set cost:", cost_validation
        
        
    gd_training = df(xm6_tr,ym6_tr,theta_training)
        # print "gd_training:", gd_training
        
    alpha = 0.00000000001
    #    print "choose alpha = ",alpha
        
    theta0_training = np.zeros((1,1025))                                
    
    #   FIND NEW THETA FOR EACH TRAINING SET SIZES    
    theta_training = grad_descent(f,df,xm6_tr,ym6_tr,theta0_training,alpha)
    #    print "theta for training set :",theta_training
    theta_training_sizes.append(theta_training)

    pred_y_training = predict_gender(xm6_tr,ym6_tr, theta_training)
    accuracy_training = get_accuracy(pred_y_training,ym6_tr)
    #    print "accuracy training:", accuracy_training

    # use theta_training got from gd (training set)
    pred_y_validation = predict_gender(xm6_vali,ym6_vali, theta_training)
    accuracy_validation = get_accuracy(pred_y_validation,ym6_vali)
    print "accuracy validation:", accuracy_validation
    print "\n"
    accu_tr.append(accuracy_training)
    accu_vali.append(accuracy_validation)
print 'accurracy_training for diff sizes list:', accu_tr
print 'accurracy_validation for diff sizes list:', accu_vali



# ******************************************************************************

act_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
# 1 1 1 1 0 0 0
# zero for female, one for male

# flip zeros and ones
def get_y_matrix_test6(row, column):
    y0 = np.ones((row,column))
    y1 = np.zeros((row,column))
    y_matrix = np.vstack((y0,y1))
    return array(y_matrix)
#print y_matrix

def get_diff_y_test6(row,coln):
    ym6 = get_y_matrix_test6(row*3,coln)
    return ym6


size6t = []
accu_tr6t = []
accu_vali6t = []
cost_training_sizes6t = []
cost_validation_sizes6t = []
# use theta found above in act_test
        
for i in range (1, 21):
    num = 5*i
    print num
    size6t.append(num)
    myfilepath = "resized"    
    act6_tr6t, act6_vali6t, act6_ts6t = get_diff_set(num, 10, 10, act_test)
        
    xm6_tr6t = get_x_matrix(act6_tr6t)
    xm6_vali6t = get_x_matrix(act6_vali6t)
    
    # flip zeros and ones    
    ym6_tr6t = get_diff_y_test6(num,1)
    ym6_vali6t = get_diff_y_test6(10,1)
    
    # -------
    pred_y_training6t = predict_gender(xm6_tr6t,ym6_tr6t, theta_training_sizes[i-1])
    accuracy_training6t = get_accuracy(pred_y_training6t,ym6_tr6t)
    #    print "accuracy training:", accuracy_training

    # use theta_training got from gd (training set from previous for loop)
    pred_y_validation6t = predict_gender(xm6_vali6t,ym6_vali6t, theta_training_sizes[i-1])
    accuracy_validation6t = get_accuracy(pred_y_validation6t,ym6_vali6t)
    print "accuracy validation:", accuracy_validation6t
    print "\n"
    accu_tr6t.append(accuracy_training)
    accu_vali6t.append(accuracy_validation6t)
print 'accurracy_training for diff sizes list:', accu_tr6t
print 'accurracy_validation for diff sizes list:', accu_vali6t

# ******************************************************************************
#        OVERFITTING AND PLOTTING

plt.figure(2)

plt.subplot(211)
plt.plot(size,accu_tr,'k',size,accu_vali,'bo')

plt.xlabel('Training set sizes')
plt.ylabel('performance')
plt.title(' accuracy vs sizes')
plt.grid(True)
plt.savefig("part_5.jpg")


plt.subplot(212)
plt.plot(size,accu_tr6t,'r--',size, accu_vali6t,'bs')

plt.ylabel('act_ test performance')
plt.title('act_test accuracy vs sizes')
plt.grid(True)
plt.savefig("part_5_acts.jpg")
plt.show()

# ------------------------------------------------------------------------------
#                       PART 6 c

def cost_matrix(x_matrix, y_matrix, theta_matrix):
    sum = 0
    for i in range(0,len(x_matrix)):
        for j in range(0,len(theta_matrix)):
            sum +=(dot(theta_matrix[j],x_matrix[i])-y_matrix[i][j])**2
    return sum
    
def dcost_matrix(x_matrix,y_matrix,theta_matrix):
    theta_matrix_T = transpose(theta_matrix)
    dotproduct = matmul(x_matrix,theta_matrix_T)
    diff = subtract(dotproduct,y_matrix)
    diff_T = transpose(diff)
    res = multiply(2,matmul(diff_T,x_matrix))
    return res

    
# ------------------------------------------------------------------------------
#                   PART 6 d
# finite differences to check 


    
# ------------------------------------------------------------------------------
# DONT USE THIS FUNCTION

#def dj(x_matrix, y_matrix, theta_matrix):
#    print x_matrix.shape
#    print y_matrix.shape
#    print theta_matrix.shape
#    djff = 2*dot((dot(theta_matrix,x_matrix.T) - y_matrix.T),(x_matrix))
#    return djff
    
#djff = dj(xm6_tr,y,theta)
# ------------------------------------------------------------------------------

print 'finite difference:'

h = 0.001
h1 = h+1


fin_y_test = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
fin_theta = [[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]]
fin_x_test = [[2,6,4,1,5],[2,4,1,5,4],[2,3,2,5,4],[1,5,2,3,2]]
dcost = dcost_matrix(fin_x_test,fin_y_test,fin_theta)
cost = cost_matrix(fin_x_test,fin_y_test,fin_theta)

print 'test run 1'
theta_ht = deepcopy(fin_theta)
d_theta = theta_ht[1][0]*h
theta_ht[1][0] = theta_ht[1][0]*h1
cost_h = cost_matrix(fin_x_test,fin_y_test,theta_ht)
dcost_dtheta = (cost_h - cost)/d_theta
print 'Finite difference',dcost_dtheta
print 'Actual derivative',dcost[1][0]


print 'test run 2'
theta_ht = deepcopy(fin_theta)
d_theta = theta_ht[2][0]*h
theta_ht[2][0] = theta_ht[2][0]*h1
cost_h = cost_matrix(fin_x_test,fin_y_test,theta_ht)
dcost_dtheta = (cost_h - cost)/d_theta
print 'Finite difference',dcost_dtheta
print 'Actual derivative',dcost[2][0]


print 'test run 3'
theta_ht = deepcopy(fin_theta)
d_theta = theta_ht[2][2]*h
theta_ht[2][2] = theta_ht[2][2]*h1
cost_h = cost_matrix(fin_x_test,fin_y_test,theta_ht)
dcost_dtheta = (cost_h - cost)/d_theta
print 'Finite difference',dcost_dtheta
print 'Actual derivative',dcost[2][2]


# -------------------------------
#           PART 7


# act6 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
# NOTE THAT ACT IS THE SAME AS ACT6
# FOR X matrix use xm6_tr, xm6_vali, xm6_ts from part 5
# 
# act6_tr, act6_vali, act6_ts = get_diff_set(100, 10, 10, act6)
        
#xm6_tr = get_x_matrix(act6_tr)
# xm6_vali = get_x_matrix(act6_vali)

# myfilepath = "resized"  
print np.shape(xm6_tr) # 100*6 in part 5
print np.shape(xm6_vali) # 10*6 in part 5

  

# 'Fran Drescher'       [1,0,0,0,0,0]
# 'America Ferrera'     [0,1,0,0,0,0]
# 'Kristin Chenoweth'   [0,0,1,0,0,0]
# 'Alec Baldwin'        [0,0,0,1,0,0]
# 'Bill Hader'          [0,0,0,0,1,0]
# 'Steve Carell'        [0,0,0,0,0,1]

y_tem = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
y_tem = array(y_tem)

y = []
for i in range(6):
  for j in range(100):
    y.append(y_tem[i])

y = array(y)
print y
#y = []
#for i in range(30):
#  for j in range(3):
#    y.append()
#
#y = array(y)
#y = y.T
#print y
#print y.shape

theta = zeros((6,1025))
# print theta

alpha = 0.00000000001
print "choose alpha = ",alpha

#print "find gd"
print "\n"

theta = zeros((6,1025))
theta_pre = ones((6,1025))
EPS = 1e-8
iter = 0 
iter_limit = 30000
while norm(theta - theta_pre) > EPS and iter < iter_limit:
    theta_pre = theta
    dj = multiply(alpha,dcost_matrix(xm6_tr,y,theta))
    theta = add(theta,-dj)
    iter +=1
print iter
print cost_matrix(xm6_tr,y,theta)

# ------------------------------------------------------------------------
def pred_yy(theta, set):
    py = []
    for j in range(0,len(set)):
        num = []
        for i in range(0,len(theta)):
            sum = dot(theta[i],set[j])
            num.append(sum)
        if amax(num) == num[0]:
            py.append(0)
        elif amax(num) == num[1]:
            py.append(1)
        elif amax(num) == num[2]:
            py.append(2)
        elif amax(num) == num[3]:
            py.append(3)
        elif amax(num) == num[4]:
            py.append(4)
        elif amax(num) == num[5]:
            py.append(5)
    return py
    
def acc_py(py):
    accu = 0.0
    for i in range(0,len(py)):
        if i < 10:
            if py[i] == 0:
                accu += 100.0/len(py)
        elif i < 20:
            if py[i] == 1:
               accu += 100.0/len(py)
        elif i < 30:
            if py[i] == 2:
                accu += 100.0/len(py)
        elif i < 40:
            if py[i] == 3:
                accu += 100.0/len(py)
        elif i < 50:
            if py[i] == 4:
                accu += 100.0/len(py)
        elif i < 60:
            if py[i] == 5:
                accu += 100.0/len(py)
    return accu
        
accu = acc_py(pred_yy(theta,xm6_tr))
#print 'Training set accuracy:', accu
accu_vali = acc_py(pred_yy(theta,xm6_vali))
#print 'Validation set accuracy:', accu_vali

# -------------------------------------------------------------------------------
#           PART 8
pic = delete(theta[0],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_fran_drescher.jpg")

pic = delete(theta[1],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_america_ferrera.jpg")

pic = delete(theta[2],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_kristin_chenoweth.jpg")

pic = delete(theta[3],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_alec_baldwin.jpg")

pic = delete(theta[4],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_bill_hader.jpg")

pic = delete(theta[5],0)
pic = reshape(pic,(32,32))
plt.figure(3)
plt.imshow(pic,cmap=cm.coolwarm)
plt.savefig("theta_for_steve_carell.jpg")
