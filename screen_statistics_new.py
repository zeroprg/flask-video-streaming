
# import the necessary packages
from skimage.measure import structural_similarity as ssim
import time
import cv2
import numpy as np 
 
mse_delta = 2000
ssim_delta = 0.6
#initial height of the image stored in app

def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
 
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
 
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



def dsize(img,**options):
    height, width = img.shape[:2]
    print ("Image height: {0} and width: {1} before resizing ".format( height, width) )
    original_height = options.get("original_height")
    if( original_height and original_height > 0 ):        
        ratio = height/original_height
        dsize = (int(height /ratio), int(width /ratio))
    else:
        dsize = (height, width)
    print ("Image height: {0} and width: {1} after resizing ".format( height, width) ) 
    return dsize

class Screen_statistic(object):
    """An emulated camera implementation that streams a repeated sequence of images"""


    def image_resize(self, img):
        height, width = img.shape[:2]
        if( self.initial_height > 0 ):
            self.initial_height = (self.initial_height + height)/2
            img = cv2.resize(img , dsize(img, original_height=self.initial_height), interpolation=cv2.INTER_CUBIC)
        else:
            self.initial_height = height
        return img


    def isAnySimularImage(self, images: dict, key, image):
        hashes = images[key]
        if(len(hashes) == 0 ):
            return False
        for _image in hashes:
            dim = image.shape[:2]
            if( dim[0] < 30 or dim[1] < 30 ): return True
            _dim = _image.shape[:2]
            #if(_dim[0] == dim[0] and _dim[1] == dim[1] ): return True
            # convert the image to grayscale and compute the hash
            #imageHash = dhash(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            _image = cv2.resize(_image , (dim[1],dim[0]))# , interpolation=cv2.INTER_CUBIC)
            d = mse(_image , image)
            #_imageHash = dhash(cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY))
            
            print (  dim , key, self.total_for_5min[key], self.total_for_10min[key], self.total_for_30min[key]  )
            
            #cv2.imshow("Frame", image)
            
#            dsize(_image, original_height=self.initial_height)
#            dsize(image, original_height=self.initial_height)
            if( d < mse_delta): 
                return True
#            elif ( compare_ssim(_image , image) > ssim_delta ):
#                return True
                
        return False


    def __init__(self, **options):
         self.initial_height = 0
         self.total_for_5min   = {"background": 0, "bicycle": 0, "bus": 0, "car": 0, "cat": 0,"dog": 0, "horse": 0, "motorbike": 0, "person": 0,  "train": 0}
         self.total_for_10min  = self.total_for_5min 
         self.total_for_30min  = self.total_for_5min
         self.total_for_1hour  = self.total_for_5min
         self.total_for_2hour  = self.total_for_5min
         self.total_for_4hour  = self.total_for_5min
         self.total_for_8hour  = self.total_for_5min
         self.images           = {"background": [], "bicycle": [], "bus": [], "car": [], "cat": [],"dog": [], "horse": [], "motorbike": [], "person": [],  "train": []}
         image = options.get("image")
         key   = options.get("key") 
         if( image and key ):
             self.images[key]  = [image]

   

    
    def refresh(self, key, image):
        tick = int(time.time())
        if( tick % 300 == 0 ):   self.total_for_5min[key] = 0
        if( tick % 600 == 0 ):   self.total_for_10min[key] = 0
        if( tick % 1200 == 0 ):  self.total_for_30min[key] = 0
        if( tick % 3600 == 0 ):  self.total_for_1hour[key] = 0
        if( tick % 10200 == 0 ): self.total_for_2hour[key] = 0
        if( tick % 24400 == 0 ): self.total_for_4hour[key] = 0
        if( tick % 48800 == 0 ): self.total_for_8hour[key] = 0
        if( not key in self.images ): return
        #image = self.image_resize(image)
        if(not self.isAnySimularImage(self.images, key, image) ):
            """ a new object with a new coordinates (value) come """
            if(len(self.images[key]) > 15 ): self.images[key].pop(0)
            self.images[key].append(image)
            #images[(i + 1) % len(images)][key] = image
            self.total_for_5min[key]   += 1
            self.total_for_10min[key]  += 1 
            self.total_for_30min[key]  += 1
            self.total_for_1hour[key]  += 1
            self.total_for_2hour[key]  += 1
            self.total_for_4hour[key]  += 1
            self.total_for_8hour[key]  += 1
   
    






    



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


    original = cv2.resize(original, dsize(original), interpolation=cv2.INTER_CUBIC)
    contrast = cv2.resize(contrast, dsize(contrast), interpolation=cv2.INTER_CUBIC)
    shopped  = cv2.resize(shopped,  dsize(shopped), interpolation=cv2.INTER_CUBIC)

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

	
