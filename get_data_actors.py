# template ----------------------------
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
# -------------------------------------
from rgb2gray import rgb2gray

from numpy import *
from matplotlib.pyplot import *

# from scipy import misc


# act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

act = list(set([a.split("\t")[0] for a in open("facescrub_actors.txt").readlines()]))


# 
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            
# --------------------------------------

#Note: you need to create the uncropped folder first in order 
#for this to work

uncropped_directory = "uncropped"
if not os.path.exists(uncropped_directory):
    os.makedirs(uncropped_directory)

# create a folder to store cropped_filename
cropped_directory = "cropped"
if not os.path.exists(cropped_directory):
    os.makedirs(cropped_directory)

# create a folder to store grayscaled images
grayscale_directory = "grayscaled"
if not os.path.exists(grayscale_directory):
    os.makedirs(grayscale_directory)

# create a folder to store resized images
resize_directory = "resized"
if not os.path.exists(resize_directory):
    os.makedirs(resize_directory)

# -----------------------------------------------------
# change file name to find actresses later

# check if the name a is the one in the list
for a in act:
    name = a.split()[1].lower()
    # print name ##
    i = 0
    for line in open("facescrub_actors.txt"):
        # filter the person's lines in the txt file
        if a in line:
            # get uncropped images -> still have duplicates
            filename = name + str(i) + '.'+ line.split()[4].split('.')[-1] 
            
            cropped_filename = filename.split('.')[0] + "_cropped" + '.' + line.split()[4].split('.')[-1]
            
            grayscaled_filename = cropped_filename.split('.')[0] + "_grayscaled"+'.'+line.split()[4].split('.')[-1]
            
            resized_filename = grayscaled_filename.split('.')[0] + "_resized"+'.'+line.split()[4].split('.')[-1]
# ---------------------------
            try:
                # coordinates where to crop images
                x1 = int(line.split()[5].split(',')[0])
                y1 = int(line.split()[5].split(',')[1])
                x2 = int(line.split()[5].split(',')[2])
                y2 = int(line.split()[5].split(',')[3])
            except IndexError:
                print "image is too small"+filename
                pass
          #  original = misc.face()
           # misc.imsave( "uncropped/"+filename, original )
            
# ------------------------------------------------------             
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename): this will retrieve the file to "uncropped/filename"
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30) 
            
            if not os.path.isfile("uncropped/"+filename):
                continue
 # ----------------------------------------------------------

            try:
                image = imread("uncropped/" + filename)
                cropped = image[y1:y2, x1:x2]
                imsave("cropped/" + cropped_filename,cropped)
            except:
                pass
        
        #    original = misc.imread( 'filename' )
         #   face = original[y1:y2, x1:x2];
            try:
                grayscaled = rgb2gray(cropped)
                imsave("grayscaled/" + grayscaled_filename,grayscaled,cmap=cm.gray)            
            except:
                pass

            try:
                resized = imresize(grayscaled,(32,32))
                imsave("resized/" + resized_filename,resized,cmap=cm.gray)
            except:
                pass
                
                
            print filename  
            print cropped_filename
            print grayscaled_filename
            print resized_filename
            print "-------------------"
            i += 1


 