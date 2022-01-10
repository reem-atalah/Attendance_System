class HaarFeature:
    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions  # White
        self.negative_regions = negative_regions  # Black

    def get_haar_feature_value(self, integralImage,scale=1):
        """
        Compute the value of a rectangle feature(x,y,w,h) at the integral image
        each haar feature is divided into 2 symmetric regions: black and white,
        the sub of the black - white regions is the value of this haar feature
        """
        ## to get the white region value
        pos_region_sum = sum([rectangle.get_region_sum(integralImage,scale) for rectangle in self.positive_regions])
        ## to get the black region value
        neg_region_sum = sum([rectangle.get_region_sum(integralImage,scale) for rectangle in self.negative_regions])
        ## getting the feature value
        return neg_region_sum - pos_region_sum

