# Finding Lane Lines on the Road

## Goal

The goal of this project is to create a software pipeline that will process
images of the road and find lane lines on the left and right of the car.

This project should deal with cases where lane lines are continuous and/or
intermittent.

## Results

For the three videos provided we get stable detection of the lane lines, *with
one notable exception* in the "challenge" video where for a short duration the
left lane is not detected.

[//]: # (Image References)

[image1]: ./images/side-by-side.png "Side by Side View"

---


## Implementation Details

### Pipeline Overview

The pipeline consists of 7 stages:

1. Convert the image to greyscale.
2. Apply a gaussian blur to mitigate the effect of noise.
3. Apply the canny edge algorithm.
4. Apply a "region of interest" mask to eliminate undesired details.
5. Use the hough algorithm to detect visible lines in the image.
6. Search the lane lines using the lines detected in stage 5 (RANSAC style).
7. Merge the original image with a drawing of the lane lines that have been found.

### Search of the Lane Lines

The most complex stage of the pipeline is stage 6 *Search the lane line*. Here is
a summary of the way it works:

* All lines that are *almost horizontal* are ignored.
* All lines that are *ascending from left to right* are marked as candidates for the *left lane line*.
* All lines that are *descending from left to right* are marked as candidates for the *right lane line*.
* For all candidates for the *left lane line*:
  1. Sum the length of all the lines that are closely aligned with it.
  2. Select the set of lines that has the highest total summed length.
* We repeat the same operation for the *right lane line* candidates.
* Using the selected *left lane line* candidates:
  * Compute a least square linear approximation of this set of lines using numpy's polyfit.
* Same thing with with the *right lane line* candidate.

The result of this process should be two lane lines that can be overlaid on the initial image.

### Parametrization

The code has been organized so that the key parameters for each stage of the pipeline
a grouped in one single data-structure which is used as a parameter.

```python
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
```

As can be seen the shape of the region of interest has also been made a parameter of the
processing pipeline which made it easier during the test phase to experiment with different
configurations.

### Tuning and Debug

To help with the tuning of the algorithm parameters and with debugging the
code has been modified to enable showing the output of the processing at any
stage of the pipeline.

This functionality makes it easy to visualize any stage of the pipeline, and observe the
effect of the parameter change.

In addition as most of this work was done natively on Linux I was also able to combine
two images side by side using the following operation:

```python
params_left = Params()
params_right = Params()

def process_image(image):
    left_side = process_image_one_side(image, params_left)
    right_side = process_image_one_side(image, params_right)
    combined_image = np.hstack((left_side, right_side))
    return combined_image

...

params_left.run_to_stage = 4
params_right.run_to_stage = 6
white_clip = clip1.fl_image(process_image)
```

This enabled two useful use-cases:

* Visualize simultaneously two different stages of the same pipeline.
* Visualize simultaneously the same stage of two pipelines using different parameter values.

![Side by Side Debug View][image1]

### Turning Road

To improve the quality and stability of the lane line detection the top part of the "region
of interest" has been truncated, making it a trapezoid instead of a triangle.

This was done after noticing that looking too far ahead in the distance when the road is turning
right or left was introducing noise in the lane line detection.

### No Lane Detected

If the lane line detection is too weak a threshold defined under which 

This is based on the total length of the hough lines reported the left or right
lane line.

### Type of Lane Line

For each of the two lane lines detected the software computes the actual length of line
deetcted and based on a threshold tries to estimate if the line is dotted or solid, which
could be useful information to have when doing vehicle localization.

The labeling of the type of line has been added to the resulting video.

## Implementation Issues

### Areas of low contrast

In a part of the "challenge" video a change in both lighting and color of the road causes my
implementation to not detect the line on the left side of the road.

One option to consider to overcome this would be to leverage the lane line color (yellow in this case),
to improve the detection algorithm.

### Areas with Sudden Color Changes or Strong Shadows

In several places we had to add filtering to avoid issues created by sudden changes in the road
color, or some strong shadows.

Not all issues are solved though as the lane line can still be jumpy in places in the "challenge"
video.

### Outlier Elimination Algorithm Complexity

As it is implemented right now the lane line detection algorithm is trying to *eliminate outliers*
from the set of the hough lines.

But the way this is currently done is not smart, and has a complexity which is in N^2 of the number
of lines detected.

This should be replaced with a RANSAC style algorithm to control the complexity.

## Failed Experiments

### Increase of Contrast

Some experiments were made to try to improve the detection of lines when the lighting conditions
were not favorable.

But this lead to many more issues as noise was amplified, and generally many more parasite hough
lines were detected.

The result of this experiment was that the lane line detection algorithm could be overwhelmed by
outliers.

## Potential Improvements

### Adaptative Parameters

In this project the parameters of each processing pipeline stage have been manually
until it produced satisfying results.

But as is demonstrated by the "challenge" video some conditions of road color and lighting
can cause the algorithm to fail.

One line of study would be to figure if some parameters can be adjusted at runtime

### Filtering of Spurious Lines

The software should be made to ignore lane lines if the slope of the line changes
suddenly by a significant amount from one image to the next.

It is a reasonably safe assumption to make that only a small rate of change can occur between
two consecutive images.

### Temporal Filtering

Implementing some form of Kalman filtering on the result of the lane line detection could
improve the precision and stability of the detection, and possibly overcome some of the
transient detection issues.

This would probably require leveraging data coming from other sensors.
