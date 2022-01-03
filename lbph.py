import cv2, sys, numpy, os,time

from skimage.exposure import histogram

def train_lbph(images):
    images_lbp=localBinaryPattern(images)
    return images_lbp



def localBinaryPattern(images):
    (s1,_,_)=images.shape
    for i in range(s1):
        images_processed[i] = histogram(get_lbp(images_lbp[i]),nbins=256) 
    return images_processed



def get_lbp(img):
    lbp_img = numpy.zeros(img.shape)
    binary = ''
    neighberhood = numpy.array([
        [-1,  0],
        [-1,  1],
        [0,  1],
        [1,  1],
        [1,  0],
        [1, -1],
        [0, -1],
        [-1, -1]
    ])

    for i in range(1, lbp_img.shape[0] - 1):
        for j in range(1, lbp_img.shape[1] - 1):
            binary = ''
            for neighber in neighberhood :
                if img[i + neighber[0], j + neighber[1]] > img[i][j] :
                    binary += '1'
                else:
                    binary += '0'
            lbp_img[i][j] = int(binary,2)
            
    return lbp_img


def predict_lbph(input_image,recognizer,labels):
    (s1,s2)=input_image.shape
    (d1,d2)=recognizer.shape
    temp=numpy.zeros((1,s1,s2))
    temp[0]=input_image
    input_histogramed=train_lbph(temp)
    (minval,index,distance)=(10000,0,0)
    for i in range(d1):
        distance=numpy.linalg.norm(recognizer[i,:]-input_histogramed)
        if distance<minval:
            index=i
            minval=distance
    return (labels[index],minval)
