
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
# from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import shutil
from shutil import copy





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


#Note: you need to create the uncropped folder first in order
#for this to work

def part1(file):
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                else:
                    try:
                        x1, y1, x2, y2 = line.split()[5].split(",");
                        im = imread("uncropped/" + filename);
                        face = im[int(y1):int(y2), int(x1):int(x2)]
                        image = imresize(face, [32, 32])
                        imsave("cropped/" + filename, rgb2gray(image), cmap = cm.gray)
                    except:
                        continue
                print filename
                i += 1
    
def part2():
    trainning_set = 100
    test_set = 10
    validation_set = 10
    
    folder = os.listdir("cropped/")
    
    for image in folder:
        f_name = image.split(".")[0]
        n_list = [v for v in f_name if v.isalpha()]
        name = f_name[0:len(n_list)]
        
        if not os.path.exists(name + "/"):
            os.mkdir(name + "/")
            os.mkdir(name + "/trainning_set/")
            os.mkdir(name + "/test_set/")
            os.mkdir(name + "/validation_set/")
        
        try:
            val = int(f_name.split(name)[1])
        
        except:
            continue
        
        if val < trainning_set:
            copy("cropped/" + image, name + "/trainning_set/" + image)
            
        elif trainning_set <= val < trainning_set + test_set:
            copy("cropped/" + image, name + "/test_set/" + image)
        
        elif trainning_set + test_set <= val < trainning_set + test_set +\
        validation_set:
            copy("cropped/" + image, name + "/validation_set/" + image)
                
  
  
def f(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return 0.0025*sum( (y - np.dot(x, theta)) ** 2)

def df(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return -0.005 * np.dot(x.T, (y - np.dot(x, theta)))
    
def grad_descent(f, df, x, y, init_t, alpha):
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while iter < max_iter:
        t -= alpha*df(x, y, t)
        # if iter % 500 == 0:
        #     print "Iter", iter
        #     print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

# def classifier(size, t):
#     x = np.empty(shape=[0, 1024])
#     y = np.array([[1 for v in range(size/2)]])
#     y1 = np.array([[0 for p in range(size/2)]])
#     y = np.concatenate((y, y1), 1)
#     y = np.reshape(y, (size,1))
#     one = np.array([[1 for q in range(size)]])
#     one = np.reshape(one, (size,1))
#     
#     hader = os.listdir("hader/trainning_set/")
#     carell = os.listdir("carell/trainning_set/")
#     
#     for i in range(size):
#         if (i < size/2):
#             im = imread("hader/trainning_set/" + hader[i])[:,:,0]
#         else :
#             im = imread("carell/trainning_set/" + carell[i-size/2])[:,:,0]
#         im = np.reshape(im, (1, 1024))
#         x = np.concatenate((x, im), 0)
#     
# 
#     x = np.concatenate((one, x), 1)
#     new_t = grad_descent(f, df, x, y, t, 5*1e-10)
#     return new_t
    
def class_or_correct(size, set, flag): 
    '''flag = 1 :classifiy
       flag = 0 :correction
    '''

    x = np.empty(shape=[0, 1024])
    y = np.array([[1 for v in range(size/2)]])
    y1 = np.array([[0 for p in range(size/2)]])
    y = np.concatenate((y, y1), 1)
    y = np.reshape(y, (size,1))
    one = np.array([[1 for q in range(size)]])
    one = np.reshape(one, (size,1))
    
    hader = os.listdir("hader/" + set)
    carell = os.listdir("carell/" + set)
    
    for i in range(size):
        if (i < size/2):
            im = imread("hader/" + set + hader[i])[:,:,0]
        else :
            im = imread("carell/"+ set + carell[i-size/2])[:,:,0]
        im = np.reshape(im, (1, 1024))
        x = np.concatenate((x, im), 0)

    x = np.concatenate((one, x), 1)
    if flag:
        t = np.zeros([1025, 1])  
        return grad_descent(f, df, x, y, t, 5*1e-10)
    else:
        correction = 0
        expect = np.dot(x, new_t)
        
        for i in range(size):
            if i < size/2:
                if expect[i] >= 0.5:
                    correction += 1
            else: 
                if expect[i] < 0.5:
                    correction += 1
        print"cost: %.3f\n" %(f(x, y, new_t))
        print"Percentage: %.3f\n" % (correction/float(size))
    
def part3(size):
    global new_t
    new_t = class_or_correct(size, "trainning_set/", 1)
    class_or_correct(size, "trainning_set/", 0)
    class_or_correct(10, "validation_set/", 0)
    

def part4():
    part3(200)
    image1 = np.reshape(new_t[1:], (32, 32))
    # imshow(image1)
    imsave("full.jpg", image1, cmap = cm.coolwarm)
    part3(4)
    image2 = np.reshape(new_t[1:], (32, 32))
    # imshow(image2)
    imsave("small.jpg", image2, cmap = cm.coolwarm)
    
# part4()

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']


def class_or_correct2(act, size, set, flag, new_t2):
    '''flag = 1 :classifiy
       flag = 0 :correction
    '''
    t = np.zeros([1025, 1])
    x = np.empty(shape=[0, 1024])
    y = np.array([[1 for v in range(size*3)]])
    y1 = np.array([[0 for p in range(size*3)]])
    y = np.concatenate((y, y1), 1)
    y = np.reshape(y, (size*6,1))
    one = np.array([[1 for q in range(size*6)]])
    one = np.reshape(one, (size*6,1))
    
    for a in act:
        name = a.split(" ")[1].lower()
        print name
        dir = os.listdir(name + set)
        for i in range(size):
            im = imread(name + set + dir[i])[:,:,0]
            im = np.reshape(im, (1, 1024))
            x = np.concatenate((x, im), 0)
    x = np.concatenate((one, x), 1)
    if flag:
        theta = grad_descent(f, df, x, y, t, 5*1e-10)
    else:
        theta = new_t2
    correction = 0
    expect = np.dot(x, theta)
        
    for i in range(size*6):
        if i < size*3:
            if expect[i] >= 0.5:
                correction += 1
        else: 
            if expect[i] < 0.5:
                correction += 1

    print "Act: " 
    print act
    print "Set: " 
    print set
    print"cost: %.3f\n" %(f(x, y, theta))
    print"Percentage: %.3f\n" % (correction/float(size*6))
    return theta
    
def part5(size):
    new_t2 = class_or_correct2(act, size, "/trainning_set/", 1, 0)
    print new_t2
    class_or_correct2(act, 10, "/validation_set/", 0, new_t2)
    class_or_correct2(act_test, 10, "/test_set/", 0, new_t2)

    
    # print new_t

# part5(100)

#part6
def cost(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return sum( (np.dot(theta.T,x) - y) ** 2)
 
def derive(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return 2 * np.dot(x, (np.dot(theta.T, x) - y).T)
    
def vectorized_grad_descent(f, df, x, y, init_t, alpha):
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while iter < max_iter:
        t -= alpha*df(x, y, t)
        # if iter % 500 == 0:
        #     print "Iter", iter
        #     print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t


def class_or_correct3(act, size, set, flag, new_t3):
    '''flag = 1 :classifiy
       flag = 0 :correction
    '''
    labels = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    t = np.zeros([1025, 6])
    x = np.empty(shape=[0, 1024])
    y = np.empty(shape=[6, 0])
    for i in range(6):
        y1 = labels[i,:]
        y1 = np.reshape(y1, [6, 1])
        for p in range(size):
            y = np.concatenate((y, y1), 1)
    one = np.array([[1 for q in range(size*6)]])
    one = np.reshape(one, (size*6,1))
    
    for a in act:
        name = a.split(" ")[1].lower()
        print name
        dir = os.listdir(name + set)
        for i in range(size):
            im = imread(name + set + dir[i])[:,:,0]
            im = np.reshape(im, (1, 1024))
            x = np.concatenate((x, im), 0)
    x = np.concatenate((one, x), 1)
    x = x.T
    if flag:
        theta = vectorized_grad_descent(cost, derive, x, y, t, 5*1e-11)
    else:
        theta = new_t3
    correction = 0
    expect = np.dot(theta.T, x)
        
    for i in range(size*6):
        if i < size:
            if argmax(expect[:,i], axis = 0) == 0:
                correction += 1
        elif i < size*2:
            if argmax(expect[:,i], axis = 0) == 1:
                correction += 1
        elif i < size*3:
            if argmax(expect[:,i], axis = 0) == 2:
                correction += 1
        elif i < size*4:
            if argmax(expect[:,i], axis = 0) == 3:
                correction += 1
        elif i < size*5:
            if argmax(expect[:,i], axis = 0) == 4:
                correction += 1
        elif i < size*6:
            if argmax(expect[:,i], axis = 0) == 5:
                correction += 1

    print "Act: " 
    print act
    print "Set: " 
    print set
    print"cost: %.3f\n" %(cost(x, y, theta))
    print"Percentage: %.3f\n" % (correction/float(size*6))
    return theta
    
def part7(size):
    global new_t3
    new_t3 = class_or_correct3(act, size, "/trainning_set/", 1, 0)
    class_or_correct3(act, 10, "/validation_set/", 0, new_t3)



def part8(size):
    part7(size)
    theta = new_t3[1:,:]
    for i in range(6):
        x = theta[:,i]
        print x.shape
        image = np.reshape(x, (32, 32))
        imsave("image" + str(i) + ".jpg", image, cmap = cm.coolwarm)

part8(100)
# os.mkdir("uncropped/")
# os.mkdir("cropped/")
# act = list(set([a.split("\t")[0] for a in open("facescrub_actors.txt").readlines()]))
# part1("facescrub_actors.txt")
# act = list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))
# part1("facescrub_actresses.txt")
    

