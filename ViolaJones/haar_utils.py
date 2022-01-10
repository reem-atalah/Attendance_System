# %%
import time
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from progress.bar import Bar
from rectangle_region import RectangleRegion
from haar_feature import HaarFeature
from skimage.color import rgb2gray
import skimage.io as io
import numpy as np
import commonfunctions as cf # this a custom module found the commonfunctions.py

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


def build_features(img_width, img_height, shift=1, min_width=1, min_height=1):
    """
    Generate values from Haar features
    White rectangles will be substracted from black ones to get each feature
    
    Params:
    - img_width, img_height: size of the original image
    - shift, the amount of distance the window will be shifted
    - min_width, min_height: the starting size of the haar window
    """
    features = []  # [Tuple(positive(white) regions, negative(black) regions),...]

    # scale feature window using size
    for window_width in range(min_width, img_width + 1):
        for window_height in range(min_height, img_height + 1):
            # iterarte over changing position of the feature window
            # initial x coordinate of the top left of the window
            x = 0
            while x + window_width < img_width:
                y = 0
                while y + window_height < img_height:
                    # all possible Haar regions
                    immediate = RectangleRegion(x, y, window_width, window_height)  # |o|
                    right = RectangleRegion(x + window_width, y, window_width, window_height)  # | |o|
                    ## for 3 rectangle types
                    right_2 = RectangleRegion(x + window_width * 2, y, window_width, window_height)  # | | |o|
                    bottom = RectangleRegion(x, y + window_height, window_width, window_height)  # | |/|o|
                    ## for 3 rectangle types
                    bottom_2 = RectangleRegion(x, y + window_height * 2, window_width, window_height)  # | |/| |/|o|
                    bottom_right = RectangleRegion(x + window_width, y + window_height, window_width, window_height) # | |/| |o|

                    # *** 2-rectangle haar ***
                    # Horizontal |w|b|
                    if x + window_width * 2 < img_width:
                        features.append(HaarFeature([immediate], [right]))
                    # Vertical |w|b|
                    if y + window_height * 2 < img_height:
                        features.append(HaarFeature([bottom], [immediate]))

                    # *** 3-rectangle haar ***
                    # Horizontal |w|b|w|
                    if x + window_width * 3 < img_width:
                        features.append(HaarFeature([immediate, right_2], [right]))
                    # Vertical |w|b|w|
                    if y + window_height * 3 < img_height:
                        features.append(HaarFeature([immediate, bottom_2],[bottom]))

                    # *** 4-diagonal haar rectangle ***
                    if x + window_width * 2 < img_width and y + window_height * 2 < img_height:
                        features.append(HaarFeature([immediate, bottom_right], [bottom, right]))

                    y += shift ## shift window position
                x += shift ## shift window position
    return features 



def apply_features(X_integralImages, features):
    """
    Build features of all the training data (integral images)
    """
    ## X : features
    X = np.zeros((len(features), len(X_integralImages)), dtype=np.int32)
    ## each row will contain a list of features , for example:
    ## feature[0][i] is the first feature of the image of index i in the data set..
    ## y: will be kept as it is => f0=([...], y); f1=([...], y),...
    ## to display progress
    bar = Bar('Processing features', max=len(features), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
    for i, feature in bar.iter(enumerate(features)):
        ## Compute the value of feature 'i' for each image in the training set
        ## it will be Input of the classifier_i
        X[i] = list(map(lambda integralImg: feature.get_haar_feature_value(integralImg), X_integralImages))
    bar.finish()

    return X ## [[ftr0 of img0, ftr0 of img1, ...][ftr1 of img0, ftr1 of img1, ...],....]