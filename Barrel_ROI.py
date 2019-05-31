from matplotlib import pyplot as plt
# from PIL import Image
import numpy as np
from PIL import Image
from roipoly import RoiPoly

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )




def set_fig(img, title):
    fig = plt.figure()
    plt.imshow(img[:, :, 0], interpolation='nearest', cmap='Greys')
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)
    return fig

def label_barrels(img):
    '''
    Generates a mask for the barrel
    '''

    
    fig = set_fig(img, 'label barrels')
    # Let user draw first ROI to mark red barrell
    roi_barrel = RoiPoly(color='r', fig=fig)
    mask_barrel = roi_barrel.get_mask(img[:, :, 0])
    print(mask_barrel.shape)
    # plt.imshow(mask_barrel)
    

    # truncated = len(img[:, :, 0].flatten())
    # x_indices = np.concatenate((img[:, :, 0].flatten().reshape(truncated,1), img[:, :, 1].flatten().reshape(truncated,1)), axis=1)
    # x_new = np.concatenate((x_indices, img[:, :, 2].flatten().reshape(truncated,1)), axis=1)
    # x_barrel = x_new[mask_barrel.flatten()]
    # x_notbarrel = x_new[np.invert(mask_barrel).flatten()]

    
    print(sum(mask_barrel.flatten()))
    # print(sum(np.invert(mask_barrel).flatten()))

    return mask_barrel
	
if __name__ == '__main__':
	img = load_image('hello.png')
	print(img.shape)
	plt.imshow(img)
	label_barrels(img)