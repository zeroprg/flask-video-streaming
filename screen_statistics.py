
# import the necessary packages
from skimage.measure import structural_similarity as ssim
import time

import numpy as np 
 
mse_delta = 1000
ssim_delta = 0.6

class Screen_statistic(Object):
    """An emulated camera implementation that streams a repeated sequence of"""

    def __init__(self,dict_of_objs):
         self.total_for_5min = {}
         self.total_for_10min  = {} 
         self.total_for_30min  = {}
         self.total_for_1hour  = {}
         self.total_for_2hour  = {}
         self.total_for_5hour  = {}
         self.total_for_10hour = {}
         self.images           = {}

    def refresh(dict_of_objs):
        tick = int(time.time())
        for key, image in  dict_of_objs:
             if( tick % 300 == 0 ):   total_for_5min[key] = 0
             if( tick % 600 == 0 ):   total_for_10min[key] = 0
             if( tick % 1200 == 0 ):  total_for_30min[key] = 0
             if( tick % 3600 == 0 ):  total_for_1hour[key] = 0
             if( tick % 10200 == 0 ): total_for_2hour[key] = 0
             if( tick % 24400 == 0 ): total_for_4hour[key] = 0
             if( tick % 48800 == 0 ): total_for_8hour[key] = 0
             if(not isAnySimularImage(imagehash,key,value) ):
                """ a new object with a new coordinates (value) come """ 
                if(len(values[key]) > 15 ): values[key].pop(0)
                values[key].append(image)
                #values[(i + 1) % len(values)][key] = image
                total_for_5min[key]   += 1
                total_for_10min[key]  += 1 
                total_for_30min[key]  += 1
                total_for_1hour[key]  += 1
                total_for_2hour[key]  += 1
                total_for_4hour[key]  += 1
                total_for_8hour[key]  += 1
   



    @staticmethod
    def isAnySimularImage(values: dict, key, image):
        hashes = values[key]
        if(len(hashes) == 0 ):
            return False
        for _image in hashes:
            if( mse(_image , image) < mse_delta): 
                return True
            elif ( ssim(_image , image) > ssim_delta ): 
                return True
        return False

'''Python Compare Two ImagesPython '''


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2

    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread("images/jp_gates_original.png")
    contrast = cv2.imread("images/jp_gates_contrast.png")
    shopped = cv2.imread("images/jp_gates_photoshopped.png")

    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread("images/jp_gates_original.png")
    contrast = cv2.imread("images/jp_gates_contrast.png")
    shopped = cv2.imread("images/jp_gates_photoshopped.png")
 
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
 
    # loop over the images
    for (i, (name, image)) in enumerate(images):
	    # show the image
	    ax = fig.add_subplot(1, 3, i + 1)
	    ax.set_title(name)
	    plt.imshow(image, cmap = plt.cm.gray)
	    plt.axis("off")
 
    # show the figure
    plt.show()
 
    # compare the images
    compare_images(original, original, "Original vs. Original")
    compare_images(original, contrast, "Original vs. Contrast")
    compare_images(original, shopped, "Original vs. Photoshopped")
