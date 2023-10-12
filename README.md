# LaneAssist

## Introduction
Algorithms for **Lane Detection** and **Lane Keeping** that were developed and used by the **[Vroom](https://vroom.web.auth.gr/)** team, where I was a member for the **2023 BFMC**. [Bosch Future Mobility Challenge](https://boschfuturemobility.com/) (BFMC), is an international technical competition initiated by Bosch Engineering Center Cluj in 2017. 
The competition invites bachelor and master student teams every year to develop **autonomous driving and connectivity** algorithms on **1:10 scale vehicles** to navigate in a designated environment simulating a **miniature smart city**. 



## Table of content
- [Introduction](#introduction)
- [Lane Detection](#lane-detection)
    - [Usage](#usage)
    - [Configuration](#configurations)
    - [How it Works](#how-it-works)
- [Lane Keeping](#lane-detection)
    - [Usage](#usage)
    - [Configuration](#configurations)
    - [How it Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)



# Lane Detection 
This class is designed to detect lanes on the road using computer vision techniques. The code is written in Python and uses the OpenCV library.

![Lane Detection-Keeping Demonstration](/gifs/result_fast.gif)

## Usage

To use the `LaneDetection` class, you can simply import it and create an instance:
```
from detect import LaneDetection

ld = LaneDetection()
```
The `lanes_detection` method takes as input an image and returns a dictionary with the following keys:

- `"frame"`: the lane detection output image (lanes, peaks and lane certainty visualization),
- `"left"`: a list of points representing the left lane,
- `"right"`: a list of points representing the right lane,
- `"left_coef"`: a 2nd degree polynomial fit of the left lane (coefficients representing the left lane),
- `"right_coef"`: same for the right lane,
- `"l_perc"`: a percentage showing the certainty of the left lane's (quantifies how much we trust it),
- `"r_perc"`: same for right lane,
- `"trust_left"`: a boolean indicating whether the left lane is trustworthy or not,
- `"trust_right"`: same for right lane,
- `"trust_lk"`: a boolean indicating whether the detected lanes are trustworthy or not.

Here's an example of how to use the `lanes_detection` method:
```
import cv2

image = cv2.imread('path/to/image')
lane_det_results = ld.lanes_detection(image)
```

## Configurations
The `LaneDetection` class uses a configuration file (`config.ini`) to set the parameters of the lane detection algorithm. You can modify the parameters in the configuration file to tweak the algorithm's behavior.

Below are the most important parameters that may need to be adjusted:
```
...
```


##  How it Works

The idea of the lane detection is to create slices at different heights, for each slice first search for lane points and then cluster the detected points into lanes.

Τhe lane detection method performes the following on a given frame : <br/>
1. Find the lane points and clusters them into lanes by using the 'peaks_detection' function. <br/>
    For each slice we run : 
    1. Points Detection method. <br/>
        The idea here is that the lane points in the slice are like squared pulses.

        ![Slice values](/image_repository/histogram_values.jpg)
    
        <!-- TODO: Histogram function implementation -->

        If we only run the point detection algorithm (without clustering) the result would be : 

        ![points detection vis](/gifs/slices_point_detection_visualization.gif)
    

    2. Cluster points into different lanes. <br/>
        After each detection we cluster the new detected points into lanes.

        ![points clustering vis](/gifs/clustering_visualization.gif)

    
    
2. Choose the correct left and right lane. 

3. Convert lane from points (list) to polyfit (array)

4. Post processing on lanes


...

# Lane Keeping
This class main focus is to create a desired lane that the car should follow and calculate a steering angle. 
...