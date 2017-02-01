
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
from PIL import Image

act = list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))

#def crop(image_array):
#    imresize(image_array)

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

#def part1():
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actresses.txt"):
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
    
#def part2():
trainning_set = 100
test_set = 10
validation_set = 10

folder = os.listdir("cropped/")

for image in filename:
    f_name = image.split(".")[0]
    n_list = [v for v in f_name if v.isalpha()]
    name = f_name[0:len(n_list)]
    
    if not os.path.exists(name + "/"):
        os.mkdir(name + "/")
    
    try:
        val = int(f_name.split(name)[1])
    
    except:
        continue
    
    if val < trainning_set:
        if not os.path.exists(n + "/trainning_set/"):
            os.mkdir(name + "/trainning_set/")
        shutil.copy("cropped/" + image, name + "/trainning_set/" + image)
        
    elif trainning_set <= val < trainning_set + test_set:
        if not os.path.exists(name + "/test_set"):
            os.mkdir(name + "/test_set/")
        
            














