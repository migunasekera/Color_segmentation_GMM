import numpy as np
# import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from scipy import misc
import imageio
from scipy.stats import multivariate_normal, describe
import math
from scipy.linalg import det,inv
from skimage.measure import find_contours, regionprops # For barrel statistics
import cv2
import glob
import os
from EM import *

def filler(imgfiles):
    '''
    Fill in for distances

    '''
    
    tmpdist = []
    tmparea = []
    for file in imgfiles[0]:

        # predict distance
        distance = os.path.basename(file)[:-4] # This removes the filename (.png), just keeps the distance
    #     if '_' in distance: # This is for the image with two distances
    #         for d in distance.split('_'):
    #             tmp.append(float(d))
    #     else:
    #         tmp.append(float(distance))
        

        try:
            tmpdist.append(float(distance))
        except ValueError:
            print(distance)
        img = cv2.imread(file)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('imgray',imgray)
        cv2.waitkey(0)
        ret, thresh = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(c) for c in contours]
        print(areas)
        

    dist = np.array(tmpdist)
    size = np.array(tmparea)
    return np.concatenate((dist,size),axis = 1)


if __name__ == '__main__':
    train_files = glob.glob('2019Proj1_train/*.png')
    test_files = glob.glob('2019Proj1_test/*.png')
    EM_files = glob.glob('EM_Output/*.png')


    data_barrel = np.load('redbarrel_data.npy')
    gauss = Model(data_barrel,3)

    filler(EM_files)


