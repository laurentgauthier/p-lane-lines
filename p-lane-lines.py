#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
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
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


class Line:
    """
    This line class is used to implement filtering of lines that should be
    kept for further processing vs lines that are to be discarded (e.g.
    almost horizontal lines), and also classify lines as candidates for
    either the left side of the road or the right side of the road.
    """
    def __init__(self, x1, y1, x2, y2):
        if x2 < x1:
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
        else:
            self.x1 = x2
            self.y1 = y2
            self.x2 = x1
            self.y2 = y1
        # Length of the line.
        self.norm = math.sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2)
        # Normalized vector directing the line.
        self.vx = (self.x2-self.x1)/self.norm
        self.vy = (self.y2-self.y1)/self.norm
        # Middle of the line segment.
        self.cx = (self.x2+self.x1)/2.0
        self.cy = (self.y2+self.y1)/2.0

    def fits(self, line, margin=15):
        vx1 = line.x1 - self.cx
        vy1 = line.y1 - self.cy
        vx2 = line.x2 - self.cx
        vy2 = line.y2 - self.cy
        d1 = (vx1 * self.vy) - (vy1 * self.vx)
        d2 = (vx2 * self.vy) - (vy2 * self.vx)
        if np.abs(d1) <= margin and np.abs(d2) <= margin:
            return True
        else:
            return False

    def is_right(self):
        return (self.y1 > self.y2)

    def is_left(self):
        return (self.y1 < self.y2)

    def keep_it(self):
        if self.vy < 0.35 and self.vy > -0.35:
            # Discard lines that are almost horizontal
            return False
        else:
            return True

    def draw(self, img, color_left=[255, 0, 0], color_right=[0, 255, 0], thickness=2):
        if not self.keep_it():
            return
        if self.is_right():
            cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color_right, thickness)
        else:
            cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color_left, thickness)

def draw_lines(line_img, lines, color_left=[255, 0, 0], color_right=[0, 255, 0], thickness=5):
    all_lines = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                all_lines.append(Line(x1,y1,x2,y2))

    # First pass of filtering and split the lines in a left and right groups.
    left_lines = []
    right_lines = []
    for line in all_lines:
        # Ignore lines that are almost horizontal
        if line.keep_it():
            # For the remaining lines partition them in left and right groups.
            if line.is_left():
                left_lines.append(line)
            else:
                right_lines.append(line)

    # Find the best set of right lane line candidates
    ref_lines = []
    ref_length = 0
    for line in right_lines:
        length = 0
        matches = [ ]
        for candidate in right_lines:
            if line.fits(candidate):
                length += candidate.norm
                matches.append(candidate)
        if length > ref_length:
            ref_length = length
            ref_lines = matches
    right_lines = ref_lines
    right_length = ref_length

    # Find the best set of left lane line candidates
    ref_lines = []
    ref_length = 0
    for line in left_lines:
        length = 0
        matches = [ ]
        for candidate in left_lines:
            if line.fits(candidate):
                length += candidate.norm
                matches.append(candidate)
        if length > ref_length:
            ref_length = length
            ref_lines = matches
    left_lines = ref_lines
    left_length = ref_length

    # Get the width and size of the image.
    height = line_img.shape[0]
    width = line_img.shape[1]

    # Draw the left line if it has been detected and is long enough.
    if right_length > 100:
        x = []
        y = []
        for line in right_lines:
            x.append(line.x1)
            x.append(line.x2)
            y.append(line.y1)
            y.append(line.y2)
        # Least square fitting.
        p = np.polyfit(y, x, 1)
        cv2.line(line_img, (int(p[0]*2*height/3+p[1]), int(2*height/3)), (int(p[0]*height+p[1]), int(height)), [255, 0, 0], 2)
        # Display a guess: is the lane line solid or dotted.
        if right_length > height*1.5:
            cv2.putText(line_img, "Solid %1.2f" % (right_length/height,), (int(width/2)+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 3)
        else:
            cv2.putText(line_img, "Dotted %1.2f" % (right_length/height,), (int(width/2)+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 3)
    else:
        # Display text indicating that no line was detected.
        cv2.putText(line_img, "NONE", (int(width/2)+30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 3)

    # Draw the right line if it has been detected and is long enough.
    if left_length > 100:
        # Average the right lines
        x = []
        y = []
        for line in left_lines:
            x.append(line.x1)
            x.append(line.x2)
            y.append(line.y1)
            y.append(line.y2)
        # Least square fitting.
        p = np.polyfit(y, x, 1)
        cv2.line(line_img, (int(p[0]*2*height/3+p[1]), int(2*height/3)), (int(p[0]*height+p[1]), int(height)), [0, 255, 0], 2)
        # Display a guess: is the lane line solid or dotted.
        if left_length > height*1.5:
            cv2.putText(line_img,"Solid %1.2f" % (left_length/height,), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
        else:
            cv2.putText(line_img,"Dotted %1.2f" % (left_length/height,), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)
    else:
        # Display text indicating that no line was detected.
        cv2.putText(line_img, "NONE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 3)

    return line_img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.4, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def color(gray):
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

# Set of parameters that used for the various stages of the processing pipeline.
class Params:
    blur_size = 7
    clahe_bypass = True
    clahe_clip = 4.0
    clahe_size = 16
    canny_low = 50
    canny_high = 150
    region_shape = [[0.06, 1.0], [0.4, 10.0/16.0], [0.6, 10.0/16.0], [0.94, 1.0]]
    hough_ro = 4
    hough_theta = 2*np.pi/180
    hough_threshold = 20
    hough_min_len = 20
    hough_max_gap = 50
    run_to_stage = 6

params_left = Params()
params_right = Params()

# Tried histogram equalization to try to deal with the road area with less contrast, but this did not help
# as the equalization happens across the full image.
# Tried CLAHE (adaptative equalization) which has more of an effect on the light area, but it is amplifying
# the noise in the light area.
def process_image(image):
    #left_side = process_image_one_side(image, params_left)
    #right_side = process_image_one_side(image, params_right)
    #combined_image = np.hstack((left_side, right_side))
    #return combined_image
    right_side = process_image_one_side(image, params_right)
    return right_side

def process_image_one_side(image, params):
    # Step 0: Convert the image from color to grey scale.
    grey_image = grayscale(image)
    if params.run_to_stage <= 0:
        return color(grey_image)
    # Step 1: Reduce the noise in the image by applying a gaussian blur.
    blurred_image = gaussian_blur(grey_image, params.blur_size)
    if params.run_to_stage <= 1:
        return color(blurred_image)
    # Step 2: Histogram equalization.
    if not params.clahe_bypass:
        #equalized_image = cv2.equalizeHist(grey_image)
        clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=(params.clahe_size,params.clahe_size))
        equalized_image = clahe.apply(blurred_image)
        if params.run_to_stage <= 2:
            return color(equalized_image)
    else:
        equalized_image = blurred_image
    # Step 3: Canny edge detection algorithm.
    canny_image = canny(equalized_image, params.canny_low, params.canny_high)
    if params.run_to_stage <= 3:
        return color(canny_image)
    height = canny_image.shape[0]
    width = canny_image.shape[1]
    # Step 4: Restrict the image to only our region of interest. Dependent on camera position.
    shape = []
    for i in range(len(params.region_shape)):
        shape.append([int(params.region_shape[i][0]*width), int(params.region_shape[i][1]*height)])
    region_image = region_of_interest(canny_image, [np.array(shape, np.int32)])
    if params.run_to_stage <= 4:
        return color(region_image)
    # Step 5: Detect lines in the image.
    hough_image = hough_lines(region_image, params.hough_ro, params.hough_theta, params.hough_threshold, params.hough_min_len, params.hough_max_gap)
    if params.run_to_stage <= 5:
        return hough_image
    # Step 6: Combine the lines and original image.
    weighted_image = weighted_img(hough_image, image)
    return weighted_image

# Video
from moviepy.editor import VideoFileClip
video_name = 'solidWhiteRight.mp4'
#video_name = 'solidYellowLeft.mp4'
#video_name = 'challenge.mp4'
white_output = video_name
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/" + video_name)
params_left.run_to_stage = 4
params_right.run_to_stage = 6
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
