class RectangleRegion:
    def __init__(self, x, y, width, height):
        # x coordinate of the start position of the region
        self.x = x
        # y coordinate of the start position of the region
        self.y = y
        # width of the rectangle region
        self.width = width
        # height of the rectangle region
        self.height = height

    def get_region_sum(self, integralImage,scale=1.0):
        ## used to get the sum of pixels in a specific region 
        ## inside an image using only 4 points of the integral image instead of all points in the region in original image
        ## using int is to assure that all values are integers
        top_left_x = int(self.x*scale)
        top_left_y = int(self.y*scale)
        bottom_right_x = top_left_x + int(self.width*scale) - 1
        bottom_right_y = top_left_y + int(self.height*scale) - 1
        sum_region=0
        sum_region = int(integralImage[bottom_right_x, bottom_right_y])
        if top_left_x > 0: sum_region -= int(integralImage[top_left_x-1, bottom_right_y])
        if top_left_y > 0: sum_region -= int(integralImage[bottom_right_x, top_left_y-1])
        if top_left_x > 0 and top_left_y > 0: sum_region += int(integralImage[top_left_x - 1, top_left_y - 1])
        return sum_region

