#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
def integeral_img(img):
    row = len(img) #the length of the row
    column = len(img) #length of the column
    integeral_img =np.zeros((row,column))
    for x  in range(row):
        for y in range(column):
            if x == 0 and y == 0: #first element in the matrix so no sum
                integeral_img[x][y] = img[x][y] #img[0][0] = 1 in example
            elif x == 0: #first row so no need to sum all the previous rows
                integeral_img[x][y] = integeral_img[x][y-1] + img[x][y]
            elif y == 0: #first column so no need to sum all the previous column
                integeral_img[x][y] = integeral_img[x-1][y] + img[x][y]
            else: # previous row + previous column - (previous column and row) + the current point 
                integeral_img[x][y] = (integeral_img[x-1][y] + integeral_img[x][y-1] - integeral_img[x-1][y-1]) + img[x][y]
    return integeral_img


# In[2]:


import enum 

class HaarlikeType(enum.Enum):
    TWO_HORIZONTAL = 0
    TWO_VERTICAL = 1
    THREE_HORIZONTAL = 2
    THREE_VERTICAL = 3
    FOUR_DIAGONAL = 4
    TYPES_COUNT = 5
    
class HaarlikeFeatureDetector:

    """This class has the functions to 
    get the haar like features
    from the integral image"""
    
    HaarWindow = [ ## width, hight === cols, rows
        #for TWO_HORIZONTAL type
        (2, 1), 
        #for TWO_VERTICAL type
        (1, 2), 
        #for THREE_HORIZONTAL type
        (3, 1), 
        #for THREE_VERTICAL type
        (1, 3), 
        #for FOUR_DIAGONAL type
        (2, 2) 
    ]
    
    def __init__(self, Img):
        """constructor takes all the required paramenters 
        required to get the haar like features"""
    
        ### setting the width of the Haar feature window
        self.width= Img.shape[1]
        ### setting the height of the feature
        self.height= Img.shape[0]
        ### setting the image that we want to get the haar like features for
        self.Img= np.array(Img)
        self.IntegralImg= integeral_img(Img)
    @staticmethod
    def getSumWindow(integralImg, x, y, w, h):
        
        """
        get sum of pixels of specific window (rectangle) inside an image
        we need to use the integral image as it simplify the 
        calculation of this sum, using the integral image, 
        instead of sum all the pixels inside the window of 
        the orginal image we just can take to 4 corners of 
        the window inside the integral images and the sum 
        will be = topLeft + bottomRight - topRight - bottomLeft
        y: staring row
        x: starting col
        w: width of the window
        h: height of the window
        """
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)
        if(x<0 or y<0 or h<0 or w<0):
            return 0
        if(y==0 or x==0):
            topLeft=0
        else:
            topLeft= integralImg[x-1,y-1]
        
        if(y==0):
            topRight=0
        else:
            topRight= integralImg[x-1+w,y-1]
            
        if(x==0):
            bottomLeft=0
        else:
            bottomLeft= integralImg[x-1,y-1+h]
            
        bottomRight= integralImg[x-1+w,y-1+h]
        return topLeft+bottomRight-topRight-bottomLeft
    
    @staticmethod
    def determineFeatures(width, height):
        """
        Determine the count of features for all types of the haarWindows
        Parameters:
        width : int
            The width of the window.
        height : int
            The height of the window.
        Returns:
        features_count : int
            The features count of this window size
        descriptions : list of shape = [features_cnt, [haartype, x, y, w, h]]
            The descriptions of each feature.
        """
        features_count = 0
        ## get the total features count for all types of haarWindows
        ## there are 5 types of haarWindows, each type has 
        ## multiple positions and multiple sizes
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            ## haarFeatureWindow width & height
            hfw_sizeX, hfw_sizeY = __class__.HaarWindow[haartype]
            ## for each position
            for x in range(0, width-hfw_sizeX+1):
                for y in range(0, height-hfw_sizeY+1):
                    #x: col
                    #y: row
                    ## for each size (starting from size= hfw_sizeX, hfw_sizeY and
                    ## increment the width and hight by hfw_sizeX, hfw_sizeY until
                    ## we reach the end of the window width & height)
                    ## the reason for the devision is to know how many 
                    ## HFW will fit in the remaining original window
                    features_count += int((width-x)/hfw_sizeX)*int((height-y)/hfw_sizeY)
        
        descriptions = np.zeros((features_count, 5))
        index = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            hfw_sizeX, hfw_sizeY = __class__.HaarWindow[haartype]
            ## for each size
            for w in range(hfw_sizeX, width+1, hfw_sizeX):
                for h in range(hfw_sizeY, height+1, hfw_sizeY):
                    ## for each position
                    for y in range(0, height-h+1):
                        for x in range(0, width-w+1):
                    
                            ## x: position(col number)
                            ## y: position(row, number)
                            ## w: width(#cols)
                            ## h: height(#rows)
                            descriptions[index] = [haartype, x, y, w, h]
                            index += 1
        
        return features_count, descriptions
    @staticmethod
    def extractFeatures(integralImg, features_descriptions):
        """
        extract the features from an image
        based on the features count and descriptions
        Returns:
        list of all haar like features in the whole image
        """
        rows, cols= integralImg.shape
        
        ## make array of features
        features = np.zeros(len(features_descriptions))
        index=0
        for desc in features_descriptions:
            features[index]=  HaarlikeFeatureDetector.getSingleFeature(integralImg, 
                                                    HaarlikeType(desc[0]),
                                                    desc[1],
                                                    desc[2],
                                                    desc[3],
                                                    desc[4],
                                                   )
            index+=1
        return features
    @staticmethod
    def getSingleFeature( integralImg, haarType, x, y, w, h):
        """
        get the featuree in a specific window position
        x= starting col
        y= starting row
        notice that x,y are flipped as the haarTypeWindows 
        dimensions is the opposite of the coordinate system
        w= width of the region
        h= height of the region
                """
        
        white_region_sum= 0
        black_region_sum= 0
        if (haarType== HaarlikeType.TWO_HORIZONTAL):
            ##negative 
            white_region_sum = HaarlikeFeatureDetector.getSumWindow (integralImg, x, y, w/2, h) 
            ##positive regions
            black_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x+ w/2, y, w/2, h) 
        elif (haarType== HaarlikeType.TWO_VERTICAL):
            ##negative 
            white_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y, w, h/2) 
            ##positive regions
            black_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y+h/2, w, h/2) 
        elif (haarType== HaarlikeType.THREE_HORIZONTAL):
            ##negative 
            white_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y, w/3, h) + HaarlikeFeatureDetector.getSumWindow (integralImg, x+ 2*w/3, y, w/3, h) 
            ##positive regions
            black_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x+ w/3, y, w/3, h) 
        elif (haarType== HaarlikeType.THREE_VERTICAL):
            ##negative 
            white_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y, w, h/3) +  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y+2*h/3, w, h/3)
            ##positive regions
            black_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y + h/3, w, h/3)
        elif (haarType== HaarlikeType.FOUR_DIAGONAL):
            white_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y, w/2, h/2)  +  HaarlikeFeatureDetector.getSumWindow (integralImg, x+w/2, y+h/2, w/2, h/2)
            black_region_sum =  HaarlikeFeatureDetector.getSumWindow (integralImg, x+w/2, y, w/2, h/2) +  HaarlikeFeatureDetector.getSumWindow (integralImg, x, y+h/2, w/2, h/2)
        return white_region_sum - black_region_sum
    
    @staticmethod
    def apply_features(features, training_data):
        """
        inputs:
        features: the o/p of determine_features[1]
        training_data: a list of tuples(IntegralImg, classification)

        Maps features onto the training dataset
        X=
        [
        [img1 ftr1, img2 ftr1, img3 ftr1],
        [img1 ftr2, img2 ftr2, img3 ftr2],
        [img1 ftr3, img2 ftr3, img3 ftr3],
        [img1 ftr4, img2 ftr4, img3 ftr4],
        [img1 ftr5, img2 ftr5, img3 ftr5],
        .
        .
        .
        ]

        """
        X = np.zeros((len(features), len(training_data)))
        ## list of classifications of the image
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for feature_description in features:
            feature_extractor = lambda intImg: HaarlikeFeatureDetector.getSingleFeature(intImg,HaarlikeType(feature_description[0]),
                                                                                  feature_description[1],
                                                                                  feature_description[2],
                                                                                  feature_description[3],
                                                                                  feature_description[4], )
            X[i] = list(map(lambda data:feature_extractor(data[0]), training_data, ))
            i += 1
        return X, y


# In[35]:


class WeakClassifier:
    
    def __init__(self, feature_description, threshold, polarity):
        """
          Args:
            feature_description: a list containing:
                feature type : 2 horizontal, 2vertical, .. etc
                the starting x ,y of the rectangle region
                rectangle region's width and height
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
         this class takes its argument(feature description) from determine_features function
        """
        self.feature_description = feature_description
        self.threshold = threshold
        self.polarity = polarity
    def classify(self, intImg):
        feature_extractor = lambda intImg: HaarlikeFeatureDetector.getSingleFeature(intImg,HaarlikeType(
                                                                                  self.feature_description[0]
                                                                                  ),
                                                                                  self.feature_description[1],
                                                                                  self.feature_description[2],
                                                                                  self.feature_description[3],
                                                                                  self.feature_description[4], )
        return 1 if self.polarity * feature_extractor(intImg) < self.polarity * self.threshold else 0


# In[4]:


origMat = np.array([
                   [1,2,3,4,5],
                   [6,7,8,9,10],
                   [11,12,13,14,15],
                   [16,17,18,19,20],
                   [21,22,23,24,25],
                   ])


# In[5]:


haar= HaarlikeFeatureDetector(origMat)


# In[6]:


cnt, features= haar.determineFeatures(origMat.shape[0],origMat.shape[1] )


# In[7]:


haar.apply_features(features, [(integeral_img(origMat), 1)]) ##replace [(integeral_img(origMat), 1)] with training data


# In[33]:


clf= WeakClassifier(features[0], -6, -1)


# In[34]:


clf.classify(origMat)


# In[ ]:




