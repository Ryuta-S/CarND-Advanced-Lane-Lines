import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class WarpRegionMask:
    def __init__(self, src, dst, warped_size, vertices):
        self.src = src
        self.dst = dst
        self.warped_size = warped_size
        self.vertices = vertices

    def birdsEye(self, img):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.backed_img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, self.warped_size, flags = cv2.INTER_LINEAR)
        return warped

    def returnBirdsEye(self, warped_img):
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        return_img = cv2.warpPerspective(warped_img, M, self.backed_img_size, flags=cv2.INTER_LINEAR)
        return return_img

    # draw the region on the basis of parameter 'src'
    def draw_region(self, img):
        result = np.copy(img)
        for i in range(len(self.src)):
            result = cv2.line(result, tuple(self.src[i]), tuple(self.src[(i+1)%len(self.src)]), (255,0,0),2)
        return result

    def region_of_interest(self, img):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, self.vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
