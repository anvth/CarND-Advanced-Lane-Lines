## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/binary_and_transformed_images.png "Binary and Transformed Images"
[image2]: ./output_images/birds_eye.png "Road Transformed"
[image3]: ./output_images/birds_view_test_images.png "Road Transformed(Test Images)"
[image4]: ./output_images/color_plus_gradient.png "Color and Gradient Threshold"
[image5]: ./output_images/color_spaces.png "Color Spaces"
[image6]: ./output_images/final_pipeline_test.png "Final Test Result"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/hls_color_threshold.png "HLS Color Threshold"
[image9]: ./output_images/pt_test_image.png "Perspective Tranform(Test Image)"
[image10]: ./output_images/sobel_with_direction.png "Sobel with Direction Threshold"
[image11]: ./output_images/sobel_x.png "Sobel X"
[image12]: ./output_images/sobel_y.png "Sobel Y"
[image13]: ./output_images/sobel_xy.png "Sobel XY"
[image14]: ./output_images/test_chessboard.png "Chessboard"
[image15]: ./output_images/test_chessboard_with_corners.png "Chesboard with corners detected"
[image16]: ./output_images/undistorted_chessboard.png "Undistorted Chessboard"
[image17]: ./output_images/undistorted_test_images.png "Undistorted Test Images"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first ten code cells of the IPython notebook located in "./Advanced-Lane-Detection.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt_text][image15]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image16]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I use the above calculated `objpoints` and `imgpoints` to undistort all the sample images provided in the directory test_images. The distortion corrected images are shown below:

![alt text][image17]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. In the first step, I took a test image and converted it into all three Gray, HLS, HSV and LAB color spaces. Here's an example of my output for this step.

![alt text][image5]

Carefully looking at the above images, I decided to go ahead with HLS color space, since in those color spaces, the main component of the image, i.e, lane markings are most prominent. Also, from the first project of this term, I was already aware of the color threshold for Yellow and White color. The HLS Color threhold image is shown below:

![alt_text][image8]

The next is to calculate Gradient Threshold using Sobel. Thresholds were computed in X, Y and XY direction. The results of them are bolow:

![alt_text][image11]

I obtain the "best" results for Sobel in the X direction with thresholds values between in the interval [20,120], using a kernel size of 15 (lines are very crisp).

![alt_text][image12]

In the Y direction, the best Sobel configuration is with thresholds in the interval [20,120] and kernel size 15.

![alt_text][image13]

The next step, I calculate the gradient direction using Sobel results. I am inclined to elect interval [pi/4, pi/2] as my best configuration. Kernel size of 15x15 seems to produce the least noise.

![alt_text][image10]

We then finally combine the above threholds to generate below results.

![alt_text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in the IPython notebook.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[240, img.shape[0] - 1],
    [595, 450],
    [690, 450],
    [1150, img.shape[0] - 1]])
dst = np.float32(
    [[200, img.shape[0] - 1],
    [200, 0],
    [1000, 0],
    [1000, img.shape[0] - 1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 240, bottom_px      | 200, bottom_px        | 
| 595, 450      | 200, 0      |
| 690, 450     | 1000, 0      |
| 1150, bottom_px      | 1000, bottom_px        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the lane-line pixels, I plotted a histogram of the pixel intensities for the tranformed images.

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating radius and curvature and the position of the vehicle with respect to center can be found in the compute_lane_curvature method of the AdvancedLaneDetectorWithMemory class present in the python notebook.

This method takes Left lane line and Right lane line (which represents respectively the LaneLine instances for
the computed left and right lanes, for the given binary warped image) and returns left curvature, right curvature and offset from center, which are all measured in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step can also be found in the AdvancedLaneDetectorWithMemory class of the Python notebook. It consists of following sub-steps:

| Method Name        | Description   | 
|:-------------:|:-------------:| 
| draw_lane_lines      | Returns an image where the computed lane lines have been drawn on top of the original warped binary image        |
|draw_lane_area    |Returns an image where the inside of the lane has been colored in bright green |
|draw_lane_lines_regions |Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image |
|combine_images |Returns a new image made up of the lane area image, and the remaining lane images are overlaid as small images in a row at the top of the the new image|

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A few things I would like improve upon, includes:
- Check out more color spaces that would pick up lane lines even under dark shadow. The current implementation fails to detect lanes in the shadow region.
- produce an exponential moving average of the line coefficients of previous frames and use it when pixel detection fails
- Apply other relevant computer vision techniques not covered by this project

