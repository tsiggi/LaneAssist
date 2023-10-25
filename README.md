# LaneAssist

## Introduction
Algorithms for **Lane Detection** and **Lane Keeping** developed and utilized by the **[Vroom](https://vroom.web.auth.gr/)** team, where I was a member for the **2023 BFMC**. [Bosch Future Mobility Challenge](https://boschfuturemobility.com/) (BFMC), is an international technical competition initiated by Bosch Engineering Center Cluj in 2017. 
The competition invites bachelor and master student teams every year to develop **autonomous driving and connectivity** algorithms for **1:10 scale vehicles** to navigate in a designated environment simulating a **miniature smart city**. 

## Table of Content
- [Introduction](#introduction)
- [Lane Detection](#lane-detection)
    - [Usage](#usage)
    - [How it Works](#how-it-works)
    - [Horizontal](#horizontal-detection)
- [Lane Keeping](#lane-keeping)
    - [Usage](#usage-1)
    - [How it Works](#how-it-works-1)
    - [Change Lane](#lane-changing)
- [Configuration](#configurations)
- [Real world Results](#real-road-scenarios)
- [Contributing](#contributing)
- [License](#license)

# Lane Detection 
This class is designed to detect lanes on the road using computer vision techniques. The code is written in Python and utilizes the OpenCV library.

![Lane Detection-Keeping Demonstration](/gifs/result_fast.gif)

## Usage

To use the `LaneDetection` class, import it and create an instance:
```python
from detect import LaneDetection

ld = LaneDetection()
```

The `lanes_detection` method takes an image as input and returns a dictionary with the following keys:
- `"frame"`: the lane detection output image (lanes, peaks and lane certainty visualization),
- `"left"`: a list of points representing the left lane,
- `"right"`: a list of points representing the right lane,
- `"left_coef"`: a 2nd degree polynomial fit of the left lane (coefficients representing the left lane),
- `"right_coef"`: same for the right lane,
- `"l_perc"`: a percentage showing the certainty of the left lane (quantifying how much we trust it),
- `"r_perc"`: same for the right lane,
- `"trust_left"`: a boolean indicating whether the left lane is trustworthy or not,
- `"trust_right"`: same for the right lane,
- `"trust_lk"`: a boolean indicating whether the detected lanes are trustworthy or not.

Example of using the `lanes_detection` method:

```python
import cv2

image = cv2.imread('path/to/image')
lane_det_results = ld.lanes_detection(image)
```

## How it Works

The lane detection method in LaneAssist operates through a series of steps, ensuring accurate identification and tracking of lanes. Here's a breakdown of the process:

### 1. Lane Points Detection and Clustering

The algorithm begins by creating slices at different heights in the input frame. For each slice, the following steps are performed using the 'peaks_detection' function:

<details>
<summary>Details</summary>

For each slice, the following processes take place:

1. **Points Detection Method:**
   The lane points in a histogram (slice) are treated as "squared pulses." The algorithm detects these pulses in the form of points.
   ![Slice values](/image_repository/histogram_values.jpg)

   <!-- TODO: Implement Histogram function -->

   Running the point detection algorithm (without clustering) results in the visualization shown below:

   ![Points Detection Visualization](/gifs/slices_point_detection_visualization.gif)

2. **Cluster Points into Different Lanes:**
   After detecting points in each slice, the algorithm clusters the newly detected points into lanes. This step involves organizing the points into coherent lanes.

   ![Points Clustering Visualization](/gifs/clustering_visualization.gif)

</details>

### 2. Lane Selection
The algorithm determines the correct left and right lanes based on the clustered points' positions and characteristics.

### 3. Lane Conversion and Representation
Detected lanes, initially represented as a list of points, are converted into a polynomial fit (polyfit). This conversion simplifies the lane representation, making it easier to work with.

### 4. Post-Processing
Post-processing steps are applied to the lanes, eliminating false lane detections and ensuring accuracy.

## Horizontal Detection
A visual representation of the algorithm in action. On the left, observe real-time detections of horizontal lines highlighted in green. On the right, gain a detailed understanding of the algorithmic process, showcasing frame-by-frame analysis for accurate line identification.

![Lane Detection-Keeping Demonstration](/gifs/horizontal_detection.gif)

# Lane Keeping
This class implements the lane keeping algorithm by calculating steering angles from the detected lines.

## Usage

To use the `LaneKeeping` class, import it and create an instance:
```python
from lanekeeping import LaneKeeping

lk = LaneKeeping()
```

The `lane_keeping` method takes the results of lane detection as input and returns :
- float : the calculated steering angle,
- img : lane-keeping output image.

Example of using the `lane_keeping` method:
```python
lane_det_results = ld.lanes_detection(image)

angle, frame = lk.lane_keeping(lane_det_results)
```

##  How it Works

The lane-keeping method creates a desired lane that the car should follow and then calculates an angle based on where that line is.

Τhe lane keeping method uses a pid controller that performs the following on a given frame : 
<br/>

1. _. 
        
2. _.

...

## Lane Changing
- Manipulating the desired lane enables lane changes.
- When changing lanes, the lane on the desired side is shifted to the opposite side!

<!-- ![lk-changing lane](/image_repository/lk_resutls.jpg) -->
![lk-changing lane](/image_repository/lane_change.jpg)



# Configurations
The `LaneDetection` and `LaneKeeping` classes uses a configuration file (`config.ini`) to set the parameters of the lane detection and the lane keeping algorithms. You can modify the parameters in the configuration file to adjust the algorithm's behavior.

Below are the most important parameters that may need adjustment:
```ini
# LANE_DETECTION
slices # (int) Number of histograms used to search for lane points.
bottom_offset # (int) Height in pixels of the first slice (height - bottom_offset).
bottom_perc # (int) Height in percentage of the last slice (height - height * bottom_perc).
peaks_min_width # (int) Minimum duration of a peak (lane point). Must be less than the value of a lane's point at the last slice.
peaks_max_width # (int) Maximum duration of a peak (lane point). Should be greater than the one at the first slice.
max_allowed_width_perc # (float) Checks if a point is near a lane. Used in lane clustering.
min_peaks_for_lane # (int) Minimum number of points that a lane must have.
optimal_peak_perc # (float) Percentage of slices = number of points that a lane must have to be considered as right or left.
allowed_certainty_perc_dif # (float) Keeps only one lane if the certainty difference is greater than this parameter.

# SQUARE PULSES AT HISTOGRAM ARE LANE POINTS
square_pulses_min_height # (int) Minimum grayscale value of a pixel to be a lane point.
square_pulses_min_height_dif # (int) Minimum height of the square pulse. When 2 pixels difference exceeds this value, a peak (lane point) has started.
square_pulses_pix_dif # (int) Distance between those 2 pixels.
square_pulses_allowed_peaks_width_error # (int) Used to calculate the upper threshold of the width of a 'square pulse'.

# HORIZONTAL LINE
TODO

# LANE_KEEPING
bottom_width # (int) Half of the distance at the bottom between the 2 lanes.
top_width # (int) Same as bottom_width but for the top distance.

TODO : change those parameters dynamically.
```

# Real Road Scenarios

Explore real-world examples below to see the potential of this solution. By fine-tuning the parameters or making specific adjustments, you can achieve exceptional results. 
Experiment with the following parameters to optimize outcomes for your unique use cases.
```ini
# DETECTION
square_pulses_min_height
square_pulses_pix_dif
square_pulses_min_height_dif
# DESIRE LANE
bottom_width
top_width
```

![Halkidiki no lights](/gifs/real_world_halkidiki_no_lights.gif) 

![Petrou levanti road](/gifs/real_world_petrou_levanti_panorama.gif)
