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


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell under the heading "Camera Calibration" of the jupyter-notebook titled `P2.ipynb` located in the CarND-Advanced-Lane-Lines repository. 

I used two helper functions to achieve this task described in the second cell, namely `calibrate_camera` and `correct_for_distortion`. In the next cell I start by defining the number of corners in x and y direction in the image. As suggested in the project statement I use the values 9 and 6 for x and y direction respectively. Then I prepare "object points" which will be the (x, y, z) coordinates of the chessboard corners in the real world. Assuming the chessboard is fixed on the (x,y) plane, I can put the z variable to be '0' such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

After obtainig the `img_points` and `obj_points`, I use the helper function `calibrate_camera` to get the camera matrix (mtx) and the distortion coefficient (dist). I apply this distortion correction to the image using the `correct_for_distortion` function to obtain the undistorted image. A sample result in included in the writeup below:

![Camera Calibration](./output_images/Camera_Calibration.jpg)
Fig. Image after correcting for distortion using the distortion coefficients obtained from camera calibration method in opencv.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the same coefficients for applying distortion correction to the test images. An example is shown below:
![Distortion Correction Test Image 1](./output_images/Camera_Calibration_test_image_1.jpg)
Fig. Distortion correction applied to the test image 1.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the same jupyter-notebook (P2.ipynb), I implemented the color transform and gradient under the heading "Color transform and gradient". I have implemented this using the helper functions `abs_sobel_thresh`,`mag_thresh`,`direc_thresh`,`S_thresh` to obtain sobel_x_binary, sobel_y_binary, magnitude_binary, directional_binary, hls_s_binary images. I also use `region_of_interest` function to focus only on the likely region of the image [just like in project 1](https://github.com/aakashkardam/Finding_Lane_Lines_Udacity_Project_1). 
```python
vertices=np.array([[(550,470),
                      (760,470),
                      (1150,720),
                      (200,720)]], dtype=np.int32) 
    masked_image = region_of_interest(COMBINE_with_HLS_THRESH, vertices) # masked image obtained
```
After obtaining all the binary images, I combine them together using the `combined_threshold` function as shown in the images below:

![Sobel X and Y Thresholds](./output_images/Sobel_X_and_Y_test_image_1.jpg)
Fig. Sobel X (left) and Sobel Y (right) gradient thresholded binary image using the threshold range of 20 to 100. 
![Directional and Magnitudinal Thresholds](./output_images/Directional_and_Magnitudinal_Binary_test_image_1.jpg)
Fig. Directional (left) and Magnitudinal (right) gradient threshold binary image using threshold range of (0.7,1.3) for directional and (20,100) for magnitudinal gradient. 
![S Threshold and Combined All Thresholds](./output_images/S_Threshold_and_combining_all_test_image_1.jpg)
Fig. Image transformed to HLS color space and a S channel thresholded binary image using the range (155,255) on the left and then combining all the binary images (right).
I use the combination of all the thresholds for my further analysis.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective Transform](./output_images/Perspective_Transform_Combined_All_test_image_1.jpg)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![Polyfitted Lane Lines](./output_images/Polyfitted_Lane_Lines_test_image_1.jpg)
![Polyfitted Lane Lines](./output_images/Lane_Lines_Window_test_image_1.jpg)


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Displaying Final Result](./output_images/Displaying_Final_Result_test_image_5.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  