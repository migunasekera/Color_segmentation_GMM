from Barrel_ROI import *
import sys
import numpy as np
# import glob
import matplotlib.pyplot as plt
# from scipy import misc
import imageio as io
from scipy.stats import multivariate_normal, describe
import math
from scipy.stats import multivariate_normal
from skimage.measure import find_contours, regionprops # For barrel statistics
import cv2
import glob
# import itertools


# print(files[0])
# data = label_barrels(load_image(files[0]))
# print(data.shape)
# set_fig(load_image(files[0]),"test")


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

    
    def __init__(self, data,n, isRGB = True):
        '''
        n: # dimensions
        '''
        
        # print(combs[:100])


        # sys.exit()
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar = False)
        self.isRGB = isRGB
        # Normally, you would create this after you've created the most optimal model. In this case,
        # we did not train a model, only a gaussian function
        # self.table = self.likelihood_Lookup()

        self.dimensions = n

        # combs = self.combinations()
        # print('Creating lookup table....')
        # self.lookup = self.likelihood(combs[:50])
        # print(self.mean)
        # pdf = multivariate_normal(combs[:50].T, self.mean, self.covariance)
        # print('finished!')
        # print('hand-calculated likelihood', self.lookup[:10])
        # print('scipy-calculatedl likelihood', pdf[:10])

        



    # def combinations(self):
    #     tmp = []
    #     for i in range(256):
    #         for j in range(256):
    #             for k in range(256):
    #                 tmp.append((i,j,k))
    #     return np.array(tmp)

    # def vectorLikelihood(self,sampleValue):
    #     # Working to speed this up
    #     # samplevalue: ndarray     
        
    #     # coeff = 1/((2*math.pi)**(1.5) * det(self.covariance)**(0.5))


    #     # tmp = -1/2* (sampleValue-self.mean) * inv(self.covariance)
    #     # exponential =  np.dot(tmp,(sampleValue-self.mean).T) # Because of memory errors, I'm doing this! ### This i
    #     # # print("shape of tmp: ", tmp.shape, "Shape of exponential: ", exponential.shape)
    #     # return coeff * exponential
        
        
    def likelihood(self,sampleValue):
        '''
        Strictly a single likelihood value will be output here. This way, I can just use the multivariate gaussian function.
        
        This is a log PDF, by the way!
        '''
        return multivariate_normal.logpdf(sampleValue, self.mean, self.covariance)

    def image2mask(self,image,threshold):
        '''
        Will convert into a numpy array of the same size as the input image, as a mask. No shape morphology was done
        inputs:
            image: numpy array. This is using the trained gaussian function
            model: Model class that has the gaussian (or gaussian mixture, if I get the time to make that)
            threshold: probability value that above constitutes a red barrel color based on model
            (set now to 1.0 * 10**-7)
        '''        
        row, column, dim = image.shape
        val = img.reshape((row*column,dim))
    #     threshold = 1.0 * 10**-7 # The probability theshold I am setting --> It is a bit under the mean.
        tmp = []
        [tmp.append(self.likelihood(v)) for v in val] # Evaluating each image is very slow, because I am evaluating every pixel
        tmp = np.array(tmp)
        return (tmp > threshold).astype(int).reshape(row,column)


if __name__ == '__main__':
    
    
    files = glob.glob('2019Proj1_train/*.png')
    data_barrel = np.load('redbarrel_data.npy')
    img = cv2.imread(files[0])
    gauss = Model(data_barrel,3)
    img = io.imread(files[0])
    row,column, dim = img.shape
    threshold = 1.0 * 10**-6
    samples = img.reshape((row*column,dim))
    tmp = np.array(gauss.likelihood(samples)).reshape((row,column)) # This is a 2D array, as it should be. This is magical, by the way.
    mask = np.ma.where(tmp > threshold, 255, 0)
    # plt.imshow(mask)

    print(find_contours(mask,1))

    #todo: Create the bounding box around the area
    
