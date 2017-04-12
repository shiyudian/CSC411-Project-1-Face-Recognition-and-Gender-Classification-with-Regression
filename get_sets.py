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
# part 5
import scipy.stats



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


# ------------------------------------------------------------------------------
#                                  PART ２
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
#   validation_set.extend(one_act[101:111])
#   test_set.extend(one_act[112:122])

#---------other method with fixed random indexs
    np.random.seed(0)
    one_act = np.random.permutation(training_number+validation_number+test_number)
    
    training_ind = one_act[0:training_number]
    validation_ind = one_act[(training_number+1):(training_number+validation_number+1)]
    test_ind = one_act[(training_number+validation_number+2):(training_number+validation_number+test_number+2)]

    for num in training_ind:
        training_set.append(a_act[num])
    for num in validation_ind:
        validation_set.append(a_act[num])
    for num in test_ind:
        test_set.append(a_act[num])
 
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
        image = imread("resized/"+a,True, 'L')
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

# CALL FUNCTION -> get x_matrix  
x_matrix_training = get_x_matrix(hc_training)
x_matrix_validation = get_x_matrix(hc_validation)

y_matrix_training = get_y_matrix(100,1)     #output y_matrix shape (200,1)
y_matrix_validation  = get_y_matrix(10,1)

theta_training =   guess_theta(1, 1025)                 
# print type(theta_training)

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
#gd_validation = df(x_matrix_validation,y_matrix_validation,theta_validation)
# print "gd_validation:", gd_validation

print "find gd"
def grad_descent(f,df,x_matrix,y_matrix,init_theta,alpha):
    EPS = 1e-8 #EPS = 10*(-5) machine epsilon
    prev_theta = init_theta-10*EPS
    theta = init_theta.copy()
    max_iter = 300000000000
    iter = 0
    while norm(theta - prev_theta) > EPS and iter < max_iter:
        prev_theta = theta.copy()
        theta -= alpha*df(x_matrix,y_matrix,theta)
        if iter % 1500 == 0:
#            print "\n"
#            print "Iteration", iter
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
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.set_aspect('equal')
plt.imshow(plot_theta,cmap=cm.coolwarm)
# plt.colorbar()
plt.show()
plt.savefig('part4_theta_with_200_training_set.png')

# ------------------------------------------------------------------------------
 
# Only two images for training dataset

#def read_image(filename):
#    image = imread("resized/"+filename,True,'L')
#    return image

#hader_tra1 = read_image(hader_training[0])


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
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
# ax.set_aspect('equal')
plt.imshow(plot_theta2,cmap=cm.coolwarm)
# plt.colorbar()
plt.show()
plt.savefig('part4_theta_with_2_training_set.png')

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
def multi_sizes(act_list):
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
        
        act6_tr, act6_vali, act6_ts = get_diff_set(num, 10, 10, act_list)
        
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
    print 'accurracy_training for diff sizes list:', accur_tr
    print 'accurracy_validation for diff sizes list:', accu_vali
    return size, accur_tr, accur_vali,cost_training_sizes, cost_validation_sizes, theta_training_sizes


lsize, l_accur_tr, l_accur_vali,l_c_tr,l_c_vali,l_theta = multi_sizes(act6)




# ******************************************************************************
#

act6_tr10, act6_vali10, act6_ts10 = get_diff_set(10, 10, 10, act6)
act6_tr50, act6_vali50, act6_ts50 = get_diff_set(50, 10, 10, act6)
act6_tr100, act6_vali100, act6_ts100 = get_diff_set(100, 10, 10, act6)

xm6_tr10 = get_x_matrix(act6_tr10)
xm6_vali10 = get_x_matrix(act6_vali10)

xm6_tr50 = get_x_matrix(act6_tr50)
xm6_vali50 = get_x_matrix(act6_vali50)

xm6_tr100 = get_x_matrix(act6_tr100)
xm6_vali100 = get_x_matrix(act6_vali100)

# act6 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# *******************************************************************************

    
ym6_tr10 = get_diff_y(10,1)
ym6_vali10 = get_diff_y(1,1)

ym6_tr50 = get_diff_y(50,1)
ym6_vali50 = get_diff_y(5,1)

ym6_tr100 = get_diff_y(100,1)
ym6_vali100 = get_diff_y(10,1)



# ******************************************************************************
#        OVERFITTING AND PLOTTING

# size 10

# display thetas in an image:
fig = plt.figure()
ax = fig.add_subplot(3,1,1)

# plt.figure(figsize=(8, 6))
# ax.set_aspect('equal')
Xtra = [theta_training10, theta_training50, theta_training100]
Ytra = [accuracy_training10,accuracy_training50,accuracy_training100]

Xvali = [theta_training10,theta_training50,theta_training100]
Yvali = [accuracy_validation10, accuracy_validation50, accuracy_validation100]


plt.plot(Xtra,Ytra)
plt.plot(Xvali,Yvali)

# plt.colorbar()
plt.show()
plt.savefig('part5_plot.png')

plt.legend(loc='upper left')

# ******************************************************************************

act_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

# zero for female, one for male
# 1 1 1 1 0 0 0

act6_tr10_ts, act6_vali10_ts, act6_ts10_ts = get_diff_set(10, 1, 1, act_test)
act6_tr50_ts, act6_vali50_ts, act6_ts50_ts = get_diff_set(50, 5, 5, act_test)
act6_tr100_ts, act6_vali100_ts, act6_ts100_ts = get_diff_set(100, 10, 10, act_test)

xm6_tr10_ts = get_x_matrix(act6_tr10_ts)
xm6_vali10_ts = get_x_matrix(act6_vali10_ts)

xm6_tr50_ts = get_x_matrix(act6_tr50_ts)
xm6_vali50_ts = get_x_matrix(act6_vali50_ts)

xm6_tr100_ts = get_x_matrix(act6_tr100_ts)
xm6_vali100_ts = get_x_matrix(act6_vali100_ts)


