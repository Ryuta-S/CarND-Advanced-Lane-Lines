import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

__author__ = 'ryutaShitomi'
__version__ = '1.0'
__date__ = '2018/10'

class ColorGrad:
    """
    Class for recognizing lane lines by using color and gradient
    """

    def __init__(self, color_space):
        """
        constructor
        @param color_space: 'RGB' or 'BGR'
            If you read image using matplotlib.image, you set 'RGB'.
            If you read image using cv2, you set 'BGR'.
        """
        self.color_space = color_space


    def absSobelThresh(self, img, orient='x', sobel_kernel=3, auto_finding_thresh = False, thresh=(0, 255)):
        """
        Apply a Sobel filter to the image
        @param img: Image to which the Sobel filter  will be applied
        @param orient: Orientation to which filter processing is applied
        @param sobel_kernel: filter size
        @param auto_finding_thresh: Whether to automatically find the threshold of the binarization process
            True if you want to find it automatically
        @param thresh: Specify threshold of binarization process
            (low_thresh, high_thresh) specified

        @return: Image after applying Sobel filter
        """
        # Calculate directional gradient
        # convert image to grayscale
        if self.color_space == 'RGB':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self.color_space == 'BGR':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Incorrect parameter 'color_space'. Set parameter 'color_space'.")

        # Apply sobel
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            raise ValueError("parameter 'orient' should be 'x' or 'y'")

        # binarization
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        if auto_finding_thresh:
            half_scaled = scaled_sobel[int(scaled_sobel.shape[0]/2):, :]
            low_thresh = np.percentile(half_scaled.ravel(), 93)
            high_thresh = low_thresh + 100
            thresh = list(thresh)
            thresh[0] = low_thresh
            thresh[1] = high_thresh
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(thresh[0] <= scaled_sobel) & (scaled_sobel <= thresh[1])] = 1
        return binary_output


    def magnitudeGradient(self, image, gray_or_r = 'gray', sobel_kernel=3, auto_finding_thresh = False, thresh=(0, 255)):
        """
        Calculate gradient magnitude
        @param image: Image to which will be calculated gradient magnitude
        @param gray_or_r: Specifying an image for which gradient magnitude is calculated
            If you want to calculate grayscale, specify 'gray'.
            If you want to calculate r_channel, specify 'r'.
        @param sobel_kernel: filter size
        @param auto_finding_thresh: Whether to automatically find the threshold of the binarization process
            True if you want to find it automatically.
        @param thresh: Specify threshold of binarization process
            (low_thresh, high_thresh) specified

        @return: Image after binarization calculating gradient magnitude
        """

        # 1) Convert to grayscale
        if gray_or_r == 'gray':
            if self.color_space == 'RGB':
                convertImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif self.color_space == 'BGR':
                convertImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Incorrect parameter 'color_space'")
        elif gray_or_r == 'r':
            if self.color_space == 'RGB':
                convertImage = image[:,:,0]
            elif self.color_space == 'BGR':
                convertImage = image[:,:,2]
            else:
                raise ValueError("Incorrect parameter 'color_space'")
        else:
            raise ValueError("parameter 'gray_or_r' should be 'gray' or 'r'")
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(convertImage, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(convertImage, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
        if auto_finding_thresh:
            half_scaled = scaled_sobel[int(scaled_sobel.shape[0]/2):, :]
            mean = np.percentile(half_scaled.ravel(), 90)
            thresh = list(thresh)
            thresh[0] = mean + 5
            thresh[1] = mean + 100
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(abs_sobelxy)
        mag_binary[(thresh[0] <= scaled_sobel) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return mag_binary


    def directionGradient(self, image, gray_or_r = 'gray', sobel_kernel=3, auto_finding_thresh = False, thresh=(0, np.pi/2)):
        """
        Calculate gradient direction.
        @param image: Image to which will be calculated gradient direction
        @param gray_or_r: Specifying an image for which gradient magnitude is calculated
            If you want to calculate grayscale, specify 'gray'.
            If you want to calculate r_channel, specify 'r'.
        @param sobel_kernel: filter size
        @param auto_finding_thresh: Whether to automatically find the threshold of the binarization process
            True if you want to find it automatically.
        @param thresh: Specify threshold of binarization process
            (low_thresh, high_thresh) specified

        @return: Image after binarization calculating gradient direction
        """

        # 1) Convert to grayscale or r_channel
        if gray_or_r == 'gray':
            if self.color_space == 'RGB':
                convertImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif self.color_space == 'BGR':
                convertImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Incorrect parameter 'color_space'")
        elif gray_or_r == 'r':
            if self.color_space == 'RGB':
                convertImage = image[:,:,0]
            elif self.color_space == 'BGR':
                convertImage = image[:,:,2]
            else:
                raise ValueError("Incorrect parameter 'color_space'")
        else:
            raise ValueError("parameter 'gray_or_r' should be 'gray' or 'r'")

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(convertImage, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(convertImage, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradient
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Calculate direction of the gradient
        dir_grad = np.arctan2(abs_sobely, abs_sobelx)
        if auto_finding_thresh:
            half_dir_grad = dir_grad[int(dir_grad.shape[0]/2):,:]
            mean = np.mean(half_dir_grad.ravel())
            thresh = list(thresh)
            thresh[0] = mean - 0.2
            thresh[1] = mean + 0.2
        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(dir_grad)
        dir_binary[(thresh[0] <= dir_grad) & (dir_grad <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return dir_binary


    def gradPipeline(self, img, ksize):
        # Apply each of the thresholding functions
        gradx = self.absSobelThresh(img, orient='x', sobel_kernel=ksize, auto_finding_thresh=True)
        # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, auto_finding_thresh = True, thresh=(20, 100))
        mag_binary = self.magnitudeGradient(img, gray_or_r = 'r', sobel_kernel=9, auto_finding_thresh=True)
        dir_binary = self.directionGradient(img, gray_or_r = 'r', sobel_kernel=13, auto_finding_thresh=True)
        if self.color_space == 'RGB':
            r_channel = img[:,:,0]
        elif self.color_space == 'BGR':
            r_channel = img[:,:,2]
        sobel_r = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sobel_r = np.absolute(sobel_r)
        scaled_r = np.uint8(255*(abs_sobel_r/ np.max(abs_sobel_r)))
        half_scaled_r = scaled_r[int(scaled_r.shape[0]/2):, :]
        mean_sobel_r = np.percentile(half_scaled_r.ravel(), 93)
        low_thresh = mean_sobel_r + 0
        binary_r = np.zeros_like(scaled_r)
        binary_r[(low_thresh <= scaled_r) & (scaled_r <= 255)] = 1

        # plt.figure(figsize=(12, 9))
        # plt.subplot(221)
        # plt.imshow(gradx, cmap='gray')
        # plt.subplot(222)
        # plt.imshow(binary_r, cmap='gray')
        # plt.subplot(223)
        # plt.imshow(mag_binary, cmap='gray')
        # plt.subplot(224)
        # plt.imshow(dir_binary, cmap='gray')
        # plt.show()

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) | (mag_binary == 1) | (binary_r == 1)) & (dir_binary == 1)] = 1
        return combined


    # Edit this function to create your own pipeline.
    def colorGradPipeline(self, img, auto_finding_saturation_thresh = False, saturation_thresh=(170, 255)):
        """
        Pipeline processing to recognize lanes using saturation and gradient.
        """
        copy_img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        if self.color_space == 'RGB':
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            r_channel = img[:,:,0]
        elif self.color_space == 'BGR':
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            r_channel = img[:,:,2]
        else:
            raise ValueError("Incorrect parameter 'color_space'")
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        ksize=5
        combined = self.gradPipeline(img, ksize=ksize)
        s_binary = np.zeros_like(s_channel)
        if auto_finding_saturation_thresh:
            rows, cols = s_channel.shape
            half_s_channel = s_channel[int(rows/2):, :]
            saturation_thresh = list(saturation_thresh)
            mean = np.percentile(half_s_channel.ravel(), 95)
            saturation_thresh[0] = mean
        s_binary[(saturation_thresh[0] <= s_channel)] = 1

        yellow = np.uint8([[[255, 255, 0]]])
        # make sure that yellow value in hsv spaces
        yellow_hsv = cv2.cvtColor(yellow, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([yellow_hsv[0,0,0]-10, 50, 50])
        upper_yellow = np.array([yellow_hsv[0,0,0]+10, 255, 255])

        # Threshold the HSV image to get only  yellow color
        yellow_binary = cv2.inRange(hsv, lower_yellow, upper_yellow)
        half_r = r_channel[r_channel.shape[0]//2:, :]
        mean = np.percentile(half_r.ravel(), 25)
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(combined), combined, s_binary)) * 255
        combined_img = np.copy(img)
        combined_binary = np.zeros_like(combined)
        combined_binary[(s_binary == 1) | (combined == 1) | (yellow_binary == 1)] = 1
        combined_img[(s_binary == 1) | (combined == 1) | (yellow_binary == 1)] = [255, 255, 255]

        return color_binary, combined_binary
