
# import the necessary packages
#from skimage.measure import structural_similarity as ssim
import time
from time import localtime, strftime
import cv2
import numpy as np
import math
import dhash

SCENE_FRAMES = 5
hash_delta = 65
mse_delta = 55
#ssim_delta = 0.6
#initial height of the image stored in app


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

""" Resize image to initial height """
def image_resize(img, initial_height):
    height, width = img.shape[:2]
    if( initial_height > 0 ):
        initial_height = (initial_height + height)/2
        img = cv2.resize(img , dsize(img, original_height=self.initial_height), interpolation=cv2.INTER_CUBIC)
    else:
        initial_height = height
    return img


class Screen_statistic(object):
    """An emulated camera implementation that streams a repeated sequence of images"""


    def isAnySimularImageByHashCode(self, image_hashes, key, hash):
        dim = image.shape[:2]
        if( dim[0] < 30 or dim[1] < 30 ): return True
        hashes = image_hashes[key]
        imageHash =  hash # dhash_own(image)
        print("image_hash:" , image_hash)
        self.image_hashes[key].append(imageHash) 
        if(len(hashes) == 0 ):
            return False

        for _imageHash in hashes:
            delta = dhash.get_num_bits_different(imageHash, _imageHash)
            if( delta < hash_delta):
                #print( key, delta ) 
                return True
#            elif ( compare_ssim(_image , image) > ssim_delta ):
#                return True

        return False

# Check if any simular image by Mean Square Error where parameter is numpy array :
#  A = np.array([confidence,startX,endX, startY, endY])
    def countDifferentImagesByMSE(self, orig_classes, classes):
         for key,value in classes.items():
             if( not key in self.images_counter ): continue
             #if(self.frame_counter > SCENE_FRAMES ):
             #    self.orig_classes[key] =[] 
                 #if( len(self.orig_classes[key])>0 ): self.orig_classes[key].pop(0)            

             #Check how big the box

             #if( value[2]< 20 or value[4]< 20 ): continue
             if( not key in orig_classes):
                 orig_classes[key] = value
                 self.images_counter[key] = len(value)
                 continue


                 
             for orig_key, orig_value in orig_classes.items():
                 
           
                 if(orig_key == key ):
                     #print(orig_value)
                     #print("len(orig_value)", len(orig_value))
                     temp=[]  
                     for _value in value:
                         Break = False
                         for _orig_value in orig_value:
                             if(int(time.time()) - _orig_value[5] > SCENE_FRAMES ):
                                 orig_value.remove(_orig_value)
                                 continue
                             #mse = ((_orig_value   _value)**2).mean()
                             #mse0 = _orig_value[0] - _value[0]
                             #mse0 = mse0*mse0
                             mse1 = _orig_value[1] - _value[1]
                             #mse2 = _orig_value[2] - _value[2]
                             mse3 = _orig_value[3] - _value[3]
                             #mse4 = _orig_value[4] - _value[4]
                             #print(mse)
                             # the same
                             mse = math.sqrt((mse1*mse1 +  mse3*mse3 )/2)
                             #print(mse)
                             if(mse < mse_delta):
                                 #print(mse)
                                 #if(self.isAnySimularImageByHashCode(self.image_hashes, key,_value[5])):
                                 #the_same +=1
                                 Break = True
                                 break
                         if not Break: temp.append(_value)
                     #new_ones = len(value) - the_same
                     #self.images_counter[key] += new_ones
                     #print("temp:", temp)            
                     #print("len(temp)", len(temp))
                     
                     orig_value.extend(temp)
                     
                     #print("orig_value:", orig_value)            
                     #print("len(orig_value)", len(orig_value))
                     
                     self.images_counter[key] =  len(orig_value)




    def __init__(self, params_queue):
        self.image_hashes     = {}
        self.orig_classes     = {}
        self.frame_counter    = 0
        
    def imcrop(img, y1,y2,x1,x2): 
     
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[y1:y2, x1:x2, :]

    def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
        img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                   (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
        y1 += np.abs(np.minimum(0, y1))
        y2 += np.abs(np.minimum(0, y1))
        x1 += np.abs(np.minimum(0, x1))
        x2 += np.abs(np.minimum(0, x1))
        return img, x1, x2, y1, y2

    def dhash_own(image, hashSize=8):
            # convert the image to grayscale and compute the hash
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize the input image, adding a single column (width) so we
            # can compute the horizontal gradient
            resized = cv2.resize(image, (hashSize + 1, hashSize))
     
            # compute the (relative) horizontal gradient between adjacent
            # column pixels
            diff = resized[:, 1:] > resized[:, :-1]
     
            # convert the difference image to a hash
            return sum([(2 << i)>>1 for (i, v) in enumerate(diff.flatten()) if v])




    def refresh(self, hashes, filenames, cam):
        #print(self.frame_counter)
        if(self.frame_counter > SCENE_FRAMES ):
            self.frame_counter = 0
            self.images_counter = IMAGES
        
        ret = self.getParametersJSON(hashes, filenames, cam)
        
        return ret

    def getParametersJSON(self, hashes, filenames, cam):
        ret =[]
        for key in hashes:
            #print(images[key])
            if len(hashes[key]) == 0: continue
            trace = Trace()
            trace.name = key
            trace.cam = cam
            tm = strftime("%H:%M:%S", localtime())
            trace.filenames = filenames.get(key,[])
            trace.x = tm
            trace.y = len(hashes[key])
            trace.text = key
           
            ret.append(trace.__dict__)
            print( trace.__dict__ )
        return ret


class Trace(dict):
    def __init__(self):
        dict.__init__(self)
        self.cam = 0
        self.x = 0
        self.y = 0
        self.name = ''
        self.text = ''
        self.filenames = []  
    def toJSON(self):
            return json.dumps(self, default=lambda o: o.__dict__,
                sort_keys=True, indent=4)

