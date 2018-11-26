import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
from color_gradient import ColorGrad
from warp_and_region_mask import WarpRegionMask

__author__ = 'ryutaShitomi'
__version__ = '1.0'
__date__ = '2018/10'

class FindLane:
    """
    Class with parameters to recognize lane and calculate curvature.
    @data left_fit: Maintain coefficient of quadratic function representing left lane
        If it is not None, execute self.search_around_poly () and find the pixel in the left lane of the next frame.
    @data right_fit: Maintain coefficient of quadratic function representing right lane
        If it is not None, execute self.search_around_poly () and find the pixel in the right lane of the next frame.
    @data warp: warp should have the class 'WarpRegionMask'
    @data color_grad: color_grad should have the class 'ColorGrad'
    @data mtx,dist: The values used to correct camera distortion
    @date count:
    """
    left_fit  = None
    right_fit = None
    warp = None
    color_grad = None
    mtx = None
    dist = None
    count = 0


    def __init__(self, ym_per_pix, xm_per_pix):
        """
        constructor
        @param ym_per_pix: Size of the metric in the y direction of the image relative to the pixel
        @param xm_per_pix: Size of the metric in the x direction of the image relative to the pixel
        """
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix


    def findPoints(self):
        """
        Find chessboard points to calculate image distortion.
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints, img


    def calcDistort(self):
        """
        Calculate camera distortion.
        The value is held by the class and corrected for each frame.
        """
        objpoints, imgpoints, img = self.findPoints()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist


    def undistImage(self, img):
        """
        Correct image distortion of image.
        @param img: Image to which will be corrected.

        @return: Undistortion image.
        """
        undist_image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist_image


    def createColorGrad(self, color_space):
        """
        Create the class 'ColorGrad'.
        Use pipeline().
        """
        self.color_grad = ColorGrad(color_space)


    def createWarp(self, src, dst, warped_size, vertices):
        """
        Create the class WarpRegionMask.
        Use pipeline().
        """
        self.warp = WarpRegionMask(src, dst, warped_size, vertices)


    def fit_polynomial(self, binary_warped):
        """
        Calculate coefficients and curvature of the quadratic function
        from the viewpoint converted binary image.
        @param binary_warped: viewpoint converted binary image

        @return out_img: An image in which the inside of the lane is painted green.
        @return curverad: curvature
        @return ideal_car_positionx: Value in the middle of the lane
        """
        # Find our lane pixels first
        if (self.left_fit is None) | (self.right_fit is None) | (self.count % 10 == 0):
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        else:
            # If you calculated the lane in the previous frame, use search_around_poly ().
            leftx, lefty, rightx, righty, out_img = self.search_around_poly(binary_warped)
        self.count += 1
        ploty, left_fitx, right_fitx = self.generatePlotValue(binary_warped)



        ## Visualization ##
        # Colors in the left and right lane regions
        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        for i in range(len(ploty)):
            out_img[i, int(left_fitx[i]):int(right_fitx[i])] = [0, 255, 0]
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]


        left_fit_cr = np.polyfit(ploty*self.ym_per_pix, left_fitx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*self.ym_per_pix, right_fitx*self.xm_per_pix, 2)
        left_curverad, right_curverad  = self.measure_curvature_real(left_fit_cr, right_fit_cr, binary_warped.shape[0]*self.ym_per_pix)
        curverad = (left_curverad + right_curverad) / 2
        ideal_car_positionx = (right_fitx[-1] - left_fitx[-1]) // 2 + left_fitx[-1]


        return out_img, curverad, ideal_car_positionx


    def find_lane_pixels(self, binary_warped):
        """
        Find lane pixels.
        Add the images in the y direction and find the lane from that value.
        @param binary_warped: viewpoint converted binary image

        @return leftx: An element in the x direction representing the position
                        of the left lane pixels in the image
        @return lefty: An element in the y direction representing the position
                        of the left lane pixels in the image
        @return rightx: An element in the x derection representing the position
                        of the right lane pixels in the image
        @return righty: An element in the y direction representing the position
                        of the right lane pixels in the image
        @return out_img: image which is drawn the windows
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        ## Histogram for challenge ##
        # histogram = np.sum(binary_warped[binary_warped.shape[0]//3:, :], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 10
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

            ## Identify the nonzero pixels in x and y within the window ##
            good_left_inds =  ((win_y_low < nonzeroy) & (nonzeroy < win_y_high) &
                               (win_xleft_low < nonzerox) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((win_y_low < nonzeroy) & (nonzeroy < win_y_high) &
                               (win_xright_low < nonzerox) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))


        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def search_around_poly(self, binary_warped):
        """
        By self.letf_fit and self.right_fit, find the pixel representing the lane.
        @param binary_warped: viewpoint converted binary image

        @return leftx: An element in the x direction representing the position
                        of the left lane pixels in the image
        @return lefty: An element in the y direction representing the position
                        of the left lane pixels in the image
        @return rightx: An element in the x derection representing the position
                        of the right lane pixels in the image
        @return righty: An element in the y direction representing the position
                        of the right lane pixels in the image
        @return out_img
        """
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        margin = 80

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # within the +/- margin of our polynomial function
        left_areax = self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2]
        left_lane_inds = ((left_areax-margin) < nonzerox) & (nonzerox < (left_areax+margin))
        right_areax = self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2]
        right_lane_inds = ((right_areax-margin) < nonzerox) & (nonzerox < (right_areax+margin))

        # extract left and right line pixel position
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        return leftx, lefty, rightx, righty, out_img


    def generatePlotValue(self, binary_warped):
        """
        Generate plot value.
        Create points to plot from coefficients of quadratic curve.
        """
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        return ploty, left_fitx, right_fitx


    def measure_curvature_real(self, left_fit_cr, right_fit_cr, y_val):
        """
        Measure the real curvature.
        """
        left_A = left_fit_cr[0]
        left_B = left_fit_cr[1]
        right_A = right_fit_cr[0]
        right_B = right_fit_cr[1]
        left_curverad = ((1+(2*left_A*y_val+left_B)**2)**(1.5))/abs(2*left_A)
        right_curverad = ((1+(2*right_A*y_val+right_B)**2)**(1.5))/abs(2*right_A)

        return left_curverad, right_curverad


    def calcDiffCarPosition(self, ideal_car_pos, real_car_pos):
        """
        Calculate how much the car's position deviates from the center of the lane.
        @param ideal_car_pos: Ideal car location
        @param real_car_pos: Real car location

        @return pos: Which side of the lane is closest to?
        @return meter_diff: How far is deviated from the ideal position
        """
        diff = ideal_car_pos - real_car_pos
        pos = 'center'
        if ideal_car_pos > real_car_pos:
            pos = 'left'
        elif ideal_car_pos < real_car_pos:
            pos = 'right'

        meter_diff = np.absolute(diff * self.xm_per_pix)
        return pos, meter_diff


    def putTextInImage(self, img, curverad, meter_diff, pos):
        """
        Put text in the image.
        @param img: Image which will be put text
        @param curverad: carvature
        @param meter_diff

        @return out_img: Image which was put text.
        """
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 1
        out_img = np.copy(img)
        text_curvature = 'Radius of Curvature = {}(m)'.format(round(curverad, 2))
        text_center = 'Vechicle is {}m {} of center'.format(round(meter_diff, 3), pos)
        cv2.putText(out_img, text_curvature, (20, 80), font, font_size, (255, 255, 255),thickness=2)
        cv2.putText(out_img, text_center, (20, 130), font, font_size, (255, 255, 255), thickness=2)
        return out_img



    def pipeline(self, img):
        """
        Pipeline processing to find a lane from an input image.
        @param img: Image you want to find a lane.

        @param out_image: Image which was found a lane.
        """
        if self.warp is None:
            raise ValueError('create WarpRegionMask class using createWarp method')
        if self.color_grad is None:
            raise ValueError('create ColorGrad class using createColorGrad method')

        if (self.dist is None) | (self.mtx is None):
            self.calcDistort()

        # undistort image
        undist_img = self.undistImage(img)
        # binarization
        color_binary, combined_binary = self.color_grad.colorGradPipeline(undist_img,
            auto_finding_saturation_thresh = True)
        # region_of_interest
        region_binary = self.warp.regionOfInterest(combined_binary)
        # warp pespective
        warped_binary = self.warp.birdsEye(region_binary)
        # find lane
        warped_line, curverad, ideal_car_posx = self.fit_polynomial(warped_binary)
        # calculate car position
        pos, meter_diff = self.calcDiffCarPosition(ideal_car_posx, warped_binary.shape[1]/2)
        warped_line = np.uint8(warped_line)
        # return warp perspective
        backed_line = self.warp.returnBirdsEye(warped_line)
        out_image = cv2.addWeighted(img, 0.8, backed_line, 0.5, 0)
        # put text
        out_image = self.putTextInImage(out_image, curverad, meter_diff, pos)
        return out_image
