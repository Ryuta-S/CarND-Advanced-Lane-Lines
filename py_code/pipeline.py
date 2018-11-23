import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
from color_gradient import ColorGrad
from warp_and_region_mask import WarpRegionMask

class FindLane:
    left_fit  = None
    right_fit = None
    warp = None # warp should have the class 'WarpRegionMask'
    color_grad = None  # color_grad should have the class 'ColorGrad'
    mtx = None
    dist = None

    def __init__(self, ym_per_pix, xm_per_pix):
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

    def findPoints(self):
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
        objpoints, imgpoints, img = self.findPoints()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist


    def undistImage(self, img):
        undist_image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist_image


    def createColorGrad(self, color_space):
        self.color_grad = ColorGrad(color_space)

    def createWarp(self, src, dst, warped_size, vertices):
        self.warp = WarpRegionMask(src, dst, warped_size, vertices)


    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        if (self.left_fit is None) | (self.right_fit is None):
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        else:
            leftx, lefty, rightx, righty, out_img = self.search_around_poly(binary_warped)
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
        car_positionx = (right_fitx[-1] - left_fitx[-1]) // 2 + left_fitx[-1]


        return out_img, curverad, car_positionx


    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
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
        left_A = left_fit_cr[0]
        left_B = left_fit_cr[1]
        right_A = right_fit_cr[0]
        right_B = right_fit_cr[1]
        left_curverad = ((1+(2*left_A*y_val+left_B)**2)++(1.5))/abs(2*left_A)
        right_curverad = ((1+(2*right_A*y_val+right_B)**2)++(1.5))/abs(2*right_A)

        return left_curverad, right_curverad


    def calcDiffCarPosition(self, car_pos, center):
        diff = car_pos - center
        pos = 'center'
        if car_pos < center:
            pos = 'left'
        elif car_pos > center:
            pos = 'right'

        meter_diff = np.absolute(diff * self.xm_per_pix)
        return pos, meter_diff


    def putTextInImage(self, img, curverad, meter_diff, pos):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 1
        out_img = np.copy(img)
        text_curvature = 'Radius of Curvature = {}(m)'.format(round(curverad, 2))
        text_center = 'Vechicle is {}m {} of center'.format(round(meter_diff, 3), pos)
        cv2.putText(out_img, text_curvature, (20, 80), font, font_size, (255, 255, 255),thickness=2)
        cv2.putText(out_img, text_center, (20, 130), font, font_size, (255, 255, 255), thickness=2)
        return out_img



    def pipeline(self, img):
        if self.warp is None:
            raise ValueError('create WarpRegionMask class using createWarp method')
        if self.color_grad is None:
            raise ValueError('create ColorGrad class using createColorGrad method')

        if (self.dist is None) | (self.mtx is None):
            self.calcDistort()

        undist_img = self.undistImage(img)
        color_binary, combined_binary = self.color_grad.colorGradPipeline(undist_img,
            auto_finding_saturation_thresh = True)
        region_binary = self.warp.region_of_interest(combined_binary)
        warped_binary = self.warp.birdsEye(region_binary)
        warped_line, curverad, car_posx = self.fit_polynomial(warped_binary)
        pos, meter_diff = self.calcDiffCarPosition(car_posx, warped_binary.shape[1]/2)
        warped_line = np.uint8(warped_line)
        backed_line = self.warp.returnBirdsEye(warped_line)
        out_image = cv2.addWeighted(img, 0.8, backed_line, 0.5, 0)
        out_image = self.putTextInImage(out_image, curverad, meter_diff, pos)
        return out_image
