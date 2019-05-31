from Barrel_ROI import *
import sys
import numpy as np
# import glob
import matplotlib.pyplot as plt
# from scipy import misc
import imageio as io
from scipy.stats import multivariate_normal, describe
from skimage.draw import rectangle_perimeter
from matplotlib.patches import Rectangle
import math
from scipy.linalg import pinv
from skimage.measure import find_contours, regionprops # For barrel statistics
import glob
import os
import argparse


def labeler(filenames,objectType,colorType):
    for f in filenames:
        barrel_mask = label_barrels(load_image(f))
        savefile = '.'.join(f.split(sep='.')[:2])
        np.save('{}_{}_{}.npy'.format(savefile,objectType,colorType),barrel_mask)

# Turn mask from an image into data
def mask2data(img, mask):
    '''
    Convert a mask from an image into data.
    input and mask have to have the same size (900,1200)
    '''

    truncated = len(img[:, :, 0].flatten())
    x_indices = np.concatenate((img[:, :, 0].flatten().reshape(truncated,1), img[:, :, 1].flatten().reshape(truncated,1)), axis=1)
    x_new = np.concatenate((x_indices, img[:, :, 2].flatten().reshape(truncated,1)), axis=1)
    x_barrel = x_new[mask.flatten()]
    x_notbarrel = x_new[np.invert(mask).flatten()]

    return x_barrel, x_notbarrel

class Model:
    '''
    Describes a 3-D gaussian function

    initial assumptions:
    zero mean, uniform covariance
    '''

    
    def __init__(self, data,imgFiles, n, threshold = -14, isRGB = True):
        '''
        n: # dimensions
        '''
        
        # print(combs[:100])


        # sys.exit()
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar = False)
        self.isRGB = isRGB
        # Normally, you would create this after you've created the most optimal model. In this case,
        # I did not train a model, only a gaussian function
        # self.table = self.likelihood_Lookup()
        self.threshold = threshold

        self.dimensions = n
        tmp = []
        for img in imgFiles:
            tmp.append(self.measure_distance(img))
        val = np.array(tmp)
        # val = np.array([tmp.append(self.measure(img)) for img in imgFiles])
        self.beta = self.leastsquares_fit(val[:,0],val[:,1])

        
        
    def likelihood(self,sampleValue):
        '''
        Strictly a single likelihood value will be output here. This way, I can just use the multivariate gaussian function.
        
        This is a log PDF, by the way!
        '''
        return multivariate_normal.logpdf(sampleValue, self.mean, self.covariance)

    def makemask(self, imgfile):
        '''
        Creates the mask. Also creates the distance that it goes
        @param 
        Return:
        Image: actual
        mask
        '''
        # This is arbitrarily set - It would probably be better if we could compare to other images. It is so incredibly low too

        img = io.imread(imgfile)
        row,column, dim = img.shape
        samples = img.reshape((row*column,dim))
        tmp = np.array(self.likelihood(samples)).reshape((row,column)) # This is a 2D array, as it should be. This is magical, by the way.
        mask = np.uint8(np.where(tmp > self.threshold, 255, 0))

        return img, mask


    def measure_distance(self, file):
        img, mask = self.makemask(file)
        props = regionprops(mask)
        dist = []
        distance = os.path.basename(file)[:-4]
        return (float(distance), float(props[0].area))
    
    def area_test_measure(self,mask):
        props = regionprops(mask)
        return props[0].area

    def test_measure_centroids(self,mask):
        props = regionprops(mask)
        centY, centX = props[0].centroid
        return centX, centY


    def leastsquares_fit(self, dist,area):
        '''
        This function will approxmate the distance of something, given the area of it.
        
        inputs:
        dist: distance as np array. Convert to the inverse of the distance, this will be the model. This is dependent variable
        area: area, given as np array. This is independent variable
        
        return:
        Linear function, which is only two parameters
        '''
        y = 1 / dist
        X = area
        # Gives the prediction. Also puts the model into form that biases can be multiplied too
        A = np.vstack((area,np.ones(area.shape))).T
        
        beta = np.dot(pinv(A),y)
        
        # includes the bias term as well.
        return beta


    def leastsquares_predict(self,values):
        '''
        Given a least square model, predict what the distance is. This is directly predicting the distance measure
        Model: beta
        '''
    #     sum_squared_residual = np.sum((y - leastsquares_predict(beta,X)) ** 2)
        
        A = np.vstack((values,np.ones(values.shape))).T
        return 1 / np.dot(A,self.beta)


if __name__ == '__main__':
    
    # read files
    train_files = glob.glob('2019Proj1_train/*.png')
    test_files = glob.glob('2019Proj1_test/*.png')
    data_barrel = np.load('redbarrel_data.npy')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--threshold", type = int, help="Optional argument to change threshold value of log-likelihood. Default is at -14")
    args = parser.parse_args()
    # thresh =0 /
    if args.threshold:
        thresh = args.threshold
    else:
        thresh = -14 # log value that is used.
    
    gauss = Model(data_barrel,train_files, 3, threshold = thresh)
    
    EM_output = 'EM_output/'
    EM_train_output = 'EM_train_output'
    imgdist = []
    val = []
    
    
    


    for i,imgfile in enumerate(test_files):
        img,mask = gauss.makemask(imgfile)
        basename = os.path.basename(imgfile)
        # io.imsave(os.path.join(EM_output,basename),mask)
        centX, centY = gauss.test_measure_centroids(mask)
        props = regionprops(mask)
        minrow, mincol, maxrow, maxcol = props[0].bbox
        rect = Rectangle((mincol,minrow), maxcol-mincol,maxrow-minrow)

        contours = find_contours(mask, 0.8)

        
        # plt.add_
        area = gauss.area_test_measure(mask)
        dist = gauss.leastsquares_predict(area)
        print("ImageNo = [0{0}], CentroidX = {1}, CentroidY = {2}, Distance = {3}".format(i,centX,centY,dist))

        
        # fig, ax = plt.subplots()
        ax1 = plt.subplot(121)
        ax1.imshow(img, interpolation = 'nearest')
        ax2 = plt.subplot(122)
        ax2.imshow(mask, interpolation='nearest', cmap= plt.cm.gray)
        plt.show()

    
