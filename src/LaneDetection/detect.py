import configparser
import math
import time
import imageio

import cv2
import numpy as np
from scipy.signal import find_peaks


class LaneDetection:
    """
    This class implements the lane detection algorithm by detecting lane lines in an input image using computer vision techniques.
    """

    def __init__(self, width, height, camera):

        self.height = height
        self.width = width

        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        self.custom_find_peaks = self.config["LANE_DETECT"].getboolean("custom_find_peaks")

        self.slices = int(self.config["LANE_DETECT"].get("slices"))
        self.print_lanes = self.config["LANE_DETECT"].getboolean("print_lanes")
        self.print_peaks = self.config["LANE_DETECT"].getboolean("print_peaks")
        self.print_lane_certainty = self.config["LANE_DETECT"].getboolean("print_lane_certainty")
        self.print_if_dashed = self.config["LANE_DETECT"].getboolean("print_if_dashed")

        self.optimal_peak_perc = float(self.config["LANE_DETECT"].get("optimal_peak_perc"))
        self.min_lane_dist_perc = float(self.config["LANE_DETECT"].get("min_lane_dist_perc"))
        self.max_lane_dist_perc = float(self.config["LANE_DETECT"].get("max_lane_dist_perc"))
        self.allowed_certainty_perc_dif = float(self.config["LANE_DETECT"].get("allowed_certainty_perc_dif"))
        self.certainty_perc_from_peaks = float(self.config["LANE_DETECT"].get("certainty_perc_from_peaks"))
        self.min_peaks_for_lane = int(self.config["LANE_DETECT"].get("min_peaks_for_lane"))

        self.extreme_coef_second_deg = float(self.config["LANE_DETECT"].get("extreme_coef_second_deg"))
        self.extreme_coef_first_deg = float(self.config["LANE_DETECT"].get("extreme_coef_first_deg"))

        self.bottom_offset = int(self.config["LANE_DETECT"].get("bottom_offset"))

        self.prev_right_lane_coeffs = None
        self.prev_left_lane_coeffs = None

        self.min_single_lane_certainty = int(self.config["LANE_DETECT"].get("min_single_lane_certainty"))
        self.min_dual_lane_certainty = int(self.config["LANE_DETECT"].get("min_dual_lane_certainty"))

        self.square_pulses_min_height = int(self.config["LANE_DETECT"].get("square_pulses_min_height"))
        self.square_pulses_pix_dif = int(self.config["LANE_DETECT"].get("square_pulses_pix_dif"))
        self.square_pulses_min_height_dif = int(self.config["LANE_DETECT"].get("square_pulses_min_height_dif"))
        self.square_pulses_allowed_peaks_width_error = int(self.config["LANE_DETECT"].get("square_pulses_allowed_peaks_width_error"))
        max_allowed_width_perc = float(self.config["LANE_DETECT"].get("max_allowed_width_perc"))
        self.max_allowed_dist = max_allowed_width_perc * self.width

        self.prev_trust_lk = False

        self.weight_for_width_distance = float(self.config["LANE_DETECT"].get("weight_for_width_distance"))
        self.weight_for_expected_value_distance = float(self.config["LANE_DETECT"].get("weight_for_expected_value_distance"))

        self.hor_step_by_step = True

        if camera == "455":
            self.choose_455()
        elif camera == "405":
            self.choose_405()
        else :
            print(">>>>> UNABLE TO SELECT CAMERA !!!")

    # ------------------------------------------------------------------------------ #

    def peaks_clustering_visualization(self, src, lanes):
        """Colours the points of the same lane the same and creates an image for 
        each slice, illustrating step by step the output of the clustering method
        that organizes points into lanes.

        Parameters
        ----------
        src : array
            Input image
        lanes : list
            A list of all lanes detected in the image, with each lane represented by
            a list of [x, y] coordinate lists.
        """
        frames = []
        last_lane_point_indexes = [0 for _ in range(len(lanes))]
        # number_of_colours = len(lanes)
        lane_colours = [(255, 0, 0),(0, 255, 0),(0,0,255),(0, 255, 255),(255, 0, 255),(255, 255, 0),(128, 0, 255),(255, 128, 0),(128, 0, 128),(0, 128, 128),(128, 128, 0)]
        
        cnt = 0
        for height in range(self.bottom_row_index, self.top_row_index - 1, self.step):
            
            # src = cv2.line(src, (0,height), (self.width,height),(60,20,220))
            src = cv2.line(src, (0,height), (self.width,height),(190,190,190))
            
            index = 0
            for lane in lanes:
                point_index = last_lane_point_indexes[index]
                point_height = -1 if point_index >= len(lane) else lane[point_index][1]

                if point_height == height :
                    
                    last_lane_point_indexes[index] += 1
                    cv2.circle(src, (lane[point_index][0], lane[point_index][1]), 2, lane_colours[index] , 2)
                
                index += 1
            
            cnt += 1
            cv2.imwrite(f"frame_{cnt}.jpg", src)


    def lanes_detection(self, src):
        """Performs the lane detection on the given frame
        1) Finds the lane peaks using the 'peaks_detection' function
        2) Choose the correct left and right lane
        3) Convert lane from points (list) to polyfit (array)
        4) Post processing on lanes

        Parameters
        ----------
        src : array
            Input image

        Results
        -------
        frame: numpy array
            Representing the lane detection output image, including the detected lanes, lane peaks, 
            and lane certainty visualization
        left: list
            Of points representing the left lane
        right: list
            Of points representing the right lane
        left_coef: numpy array
            Representing a 2nd degree polynomial fit of the left lane (coefficients representing the left lane)
        right_coef: numpy array
            Representing a 2nd degree polynomial fit of the right lane (coefficients representing the right lane)
        l_perc: float
            The percentage of points belonging to the left lane
        r_perc: float
            The percentage of points belonging to the right lane
        trust_left_lane: boolean
            Indicating whether the left lane is trustworthy or not
        trust_right_lane: boolean
            Indicating whether the right lane is trustworthy or not
        trust_lk: boolean
            Indicating whether the detected lanes are trustworthy or not
        """

        lanes, peaks = self.peaks_detection(src)
        # self.peaks_clustering_visualization(src, lanes)
        left, right = self.choose_correct_lanes(lanes)

        # Create lanes (polyfits) (vizualize lanes)
        left_coef, right_coef = self.create_lanes_from_peaks(src, left, right)

        # lanes post process
        (
            left_coef,
            right_coef,
            l_perc,
            r_perc,
            trust_l,
            trust_r,
            trust_lk,
        ) = self.lanes_post_processing(src, left_coef, left, right_coef, right)
        
        # visualize peaks
        if self.print_peaks:
            self.visualize_all_peaks(src, peaks)
            self.visualize_peaks(src, left, right)

        # dashed_l, dashed_r = self.check_for_dashed(left, right, src)

        lane_det_results = {
            "frame": src,
            "left": left,
            "right": right,
            "left_coef": left_coef,
            "right_coef": right_coef,
            "l_perc": l_perc,
            "r_perc": r_perc,
            "trust_left": trust_l,
            "trust_right": trust_r,
            # "dashed_left": dashed_l,
            # "dashed_right": dashed_r,
            "trust_lk": trust_lk,
        }

        return lane_det_results

    # ------------------------------- HORIZONTAL ----------------------------------- #

    def visualize_horizontal_line(self, frame, detected_line_segment, line):
        if detected_line_segment : 
            frame = cv2.line(frame, detected_line_segment[0], detected_line_segment[1], (143,188,143), thickness=5)

            # w1 = detected_line_segment[0][0]
            # w2 = detected_line_segment[1][0]
            # h1 = int(line["slope"] * w1 + line["intercept"])
            # h2 = int(line["slope"] * w2 + line["intercept"])

            # frame = cv2.line(frame, (w1,h1), (w2,h2), (150,150,150), thickness=3)

    def horizontal_detection(self, frame, max_allowed_slope=0.25):
        
        main_point, line = self.detect_main_point(frame, max_allowed_slope)

        detected_line_segment = self.detect_lane_line_endpoints(frame, main_point, line, max_allowed_slope)
    
        hor_min_width_dist = 0.2 * self.width
        hor_exists = detected_line_segment and (detected_line_segment[1][0] - detected_line_segment[0][0]) > hor_min_width_dist

        return detected_line_segment, line, hor_exists

    def detect_main_point(self, frame, max_allowed_slope):
        
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        max_iterations = 3

        region_of_interest_perc = 0.4

        start_of_ROI_perc = (1 - region_of_interest_perc)/2
        end_of_ROI_perc = 1 - start_of_ROI_perc
        width_start = int(self.width * start_of_ROI_perc)
        width_end = int(self.width * end_of_ROI_perc)

        top_height_perc = 0.5
        bot_height_perc = 0.95
        
        height_start = int(top_height_perc * self.height)
        height_end = int(bot_height_perc * self.height)
        
        main_point = None
        line = None

        for iterations in range(1, max_iterations):

            num_slices = 2**iterations + 1

            widths_of_slices = np.linspace(width_start, width_end, num_slices)

            # keep odd positions (indices), others (even pos) have been checked or does not matter
            widths_of_slices = widths_of_slices[1::2]

            for slice_width in widths_of_slices:
                slice_width = int(slice_width)

                tmp = [src[h][slice_width] for h in range(height_start, height_end)]
            
                ps = self.find_lane_peaks(
                    slice=tmp,
                    height_norm=0,
                    min_height=self.square_pulses_min_height,
                    min_height_dif=self.square_pulses_min_height_dif,
                    pix_dif=self.square_pulses_pix_dif,
                    allowed_peaks_width_error=self.square_pulses_allowed_peaks_width_error,
                )

                for point in ps:
                    
                    base_point = {"height": point + height_start, "width":slice_width}
                    point_l = self.search_for_near_point(frame, gray_frame=src, base_point=base_point, max_allowed_slope=max_allowed_slope, diraction="left", width_step_perc=0.015)
                    point_r = self.search_for_near_point(frame, gray_frame=src, base_point=base_point, max_allowed_slope=max_allowed_slope, diraction="right", width_step_perc=0.015)

                    num_of_slopes = 1
                    
                    # x = width, y = height
                    if point_l is not None and point_r is not None:
                        slope_l = (base_point["height"] - point_l["height"]) / (base_point["width"] - point_l["width"])
                        slope_r = (base_point["height"] - point_r["height"]) / (base_point["width"] - point_r["width"])
                        slope = (slope_l + slope_r)/2
                        num_of_slopes = 2
                    elif point_l is not None:
                        slope = (base_point["height"] - point_l["height"]) / (base_point["width"] - point_l["width"])
                    elif point_r :
                        slope = (base_point["height"] - point_r["height"]) / (base_point["width"] - point_r["width"])
                    else :
                        continue
                    
                    if abs(slope) > max_allowed_slope :
                        if self.hor_step_by_step:
                            cv2.circle(frame, (base_point["width"], base_point["height"]), 2, (0,255,0), 2)
                        continue

                    if not line or (line and num_of_slopes > line["num_of_slopes"]) or (line and num_of_slopes == line["num_of_slopes"] and abs(line['slope']) > abs(slope) ) :
                        intercept = base_point["height"] - slope * base_point["width"]
                        
                        line = {"slope": slope, "intercept": intercept, "num_of_slopes": num_of_slopes}
                    
                        main_point = base_point.copy()

                    if self.hor_step_by_step:
                        cv2.circle(frame, (base_point["width"], base_point["height"]), 2, (0,0,0), 2)

                if self.hor_step_by_step :
                    frame = cv2.line(frame,(slice_width, height_start),(slice_width, height_end),(60,20,220))

                    # cv2.imshow('hor2', frame)
                    # cv2.waitKey(0)

            if main_point :
                return main_point, line
        
        return None, None

    def detect_lane_line_endpoints(self, frame, main_point, line, max_allowed_slope):
        
        if not main_point : 
            return None
        
        left_side_points = [main_point]
        right_side_points = [main_point]

        src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_left_point = True
        while new_left_point :
            
            point_l = self.search_for_near_point(frame, gray_frame=src, base_point=left_side_points[-1], max_allowed_slope=max_allowed_slope, diraction="left", line=line)
            
            if not point_l :
                new_left_point = False
            else :
                left_side_points.append(point_l)

        new_right_point = True
        while new_right_point :    
            point_r = self.search_for_near_point(frame, gray_frame=src, base_point=right_side_points[-1], max_allowed_slope=max_allowed_slope, diraction="right", line=line)
            
            if not point_r :
                new_right_point = False
            else :
                right_side_points.append(point_r)                

        width_dif = (right_side_points[-1]["width"] - left_side_points[-1]["width"])
        if width_dif == 0 :
            return None
        
        slope = (right_side_points[-1]["height"] - left_side_points[-1]["height"]) / width_dif
       
        if abs(slope) > max_allowed_slope :
            return None

        x1 = (left_side_points[-1]['width'], left_side_points[-1]['height'])
        x2 = (right_side_points[-1]['width'], right_side_points[-1]['height'])

        return (x1,x2)

    def search_for_near_point(self, frame, gray_frame, base_point, max_allowed_slope, diraction="right", width_step_perc=0.03, line=None):
        
        max_allowed_height_dif_from_line = 0.03 * self.height
        
        operator = 1 if diraction=="right" else -1 

        width_step = int(width_step_perc * self.width)
        sliding_window_height = int(0.15 * self.height)
        
        width = int(base_point['width'] + operator* width_step)
        height = base_point['height'] if not line else line["slope"] * width + line["intercept"]
        start = int(height - sliding_window_height/2)
        end = int(height + sliding_window_height/2)

        if (width < 0 or width >= self.width) or (start < 0 or end >= self.height) :
            return None

        tmp = [gray_frame[h][width] for h in range(start, end)]
        ps = self.find_lane_peaks(
            slice=tmp,
            height_norm=0,
            min_height=self.square_pulses_min_height,
            min_height_dif=self.square_pulses_min_height_dif,
            pix_dif=self.square_pulses_pix_dif,
            allowed_peaks_width_error=self.square_pulses_allowed_peaks_width_error,
        )
        
        if self.hor_step_by_step :
            frame = cv2.line(frame,(width, start),(width, end),(60,20,220))
            # cv2.imshow('hor2', frame)
        
        point = None
        detection_height = None

        if len(ps)==1 :
            detection_height = ps[0]
        
        elif len(ps) > 1 :
            min_index = 0
            cnt = 1
            while cnt < len(ps) :
                if abs(ps[min_index] + start - base_point['height']) >= abs(ps[cnt] + start - base_point['height']) :
                    min_index = cnt
                cnt +=1 
            detection_height = ps[min_index]
        else :
            return None

        slope_with_base = abs((base_point["height"] - (detection_height + start)) / (base_point["width"] - width))
        is_height_accepted = True if not line else abs(detection_height + start - height) <= max_allowed_height_dif_from_line
       
        if slope_with_base <= max_allowed_slope and is_height_accepted:
            point = {'height': detection_height + start, "width": width}
            if self.hor_step_by_step :
                cv2.circle(frame, (point["width"], point["height"]), 2, (0,0,0), 2)
                # cv2.imshow('hor2', frame)
        else :
            if self.hor_step_by_step :
                
                cv2.circle(frame, (width, detection_height + start), 2, (0,250,0), 2)
                # cv2.imshow('hor2', frame)
        return point
    # ------------------------------------------------------------------------------ #

    def peaks_detection(self, frame):
        """Takes an image as input and returns the peaks of several different slices.
        These peaks are lane points. After finding every peak of a slice, the points
        are segmented into different lanes depending on their (x,y) values.

        Steps :
        1) Create the slices
        2) For each slice, find the peaks using a peak detection algorithm. (find_lane_peaks)
        3) Segment the peaks into different lanes based on the (x,y) values. (peaks_clustering)
        4) Return a list of lists of peaks representing the lane points in the image and a list
           of all the detected peaks.

        Parameteres
        -----------
        frame : array
            Input frame

        Returns
        -------
        lanes : list
            of lists of peaks representing different lanes
        peaks : list
            of all the detected peaks
        """

        src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        peaks = []
        lanes = []
        frames = []

        cnt = 0

        for height in range(self.bottom_row_index, self.top_row_index - 1, self.step):

            tmp = [int(x) for x in src[height]]

            height_norm = self.height_norm[cnt] if not self.is_horizontal else 0

            ps = self.find_lane_peaks(
                tmp,
                height_norm,
                self.square_pulses_min_height,
                self.square_pulses_min_height_dif,
                self.square_pulses_pix_dif,
                self.square_pulses_allowed_peaks_width_error,
            )
            if not self.is_horizontal:
                cnt += 1

            for point in ps:
                peaks.append([point, height])

            lanes = self.peaks_clustering(ps, height, lanes)

            # VISUALIZATION OF THE SLICE AND DETECTED POINTS (CREATE AN IMAGE FOR EACH SLICE)
            # frame = cv2.line(frame,(0,height),(self.width,height),(60,20,220))

            # self.visualize_all_peaks(frame, peaks, (0,0,0))
            
            # cv2.imwrite(f"frame_{cnt}.jpg", frame)

        return lanes, peaks

    def choose_correct_lanes(self, lanes):
        """Given a list of all lanes detected in an image, this function selects the left and right lanes.

        Parameters
        ----------
        lanes : list
            A list of all lanes detected in the image, with each lane represented by a list of [x, y] 
            coordinate lists.

        Returns
        -------
        left : list
            A list of [x,y] coordinate lists of the left lane.
        right : list
            A list of [x,y] coordinate lists of the right lane.
        """

        # initialize the 2 lanes
        left, right = [], []

        # Delete lanes with a set of points smaller than 3
        lanes = [lane for lane in lanes if len(lane) >= self.min_peaks_for_lane]

        # Choose left and right
        for lane in lanes:
            # check the lanes from left to right
            length = len(lane)
            # For the left side
            if lane[0][0] <= self.width / 2:
                # if lane is empty add the most right lane of the left side
                if not left:
                    left = lane
                    continue
                # left lane = the most right lane of the left side with most points (similar for the right lane)
                if length > self.slices * self.optimal_peak_perc or length > len(left):
                    left = lane
            # For the right side
            else:
                if not right:
                    right = lane
                    # if the leftest lane (of the right side) has more peaks than this WE FOUND OUR RIGHT LANE
                    if length > self.slices * self.optimal_peak_perc:
                        break
                    continue
                # here we dont want to add (or length > len(right)) cause right lane is the most left lane of the right side
                if length > self.slices * self.optimal_peak_perc:
                    right = lane
                    break

        # Check if these 2 lanes are "correct" (if they are near) (else keep only one lane)
        # if left and right:

        #     left_top, left_bot = self.calculate_lane_boundaries(left)
        #     right_top, right_bot = self.calculate_lane_boundaries(right)

        #     top_dif = (right_top - left_top) / 2
        #     bot_dif = (right_bot - left_bot) / 2

        #     # if lanes width difference is small/big delete a lane
        #     if not (
        #         top_dif > self.min_top_width_dif
        #         and top_dif < self.max_top_width_dif
        #         and bot_dif > self.min_bot_width_dif
        #         and bot_dif < self.max_bot_width_dif
        #     ):
        #         # keep lane with most peaks
        #         if len(left) > len(right):
        #             right = []
        #         else:
        #             left = []

        return left, right

    def create_lanes_from_peaks(self, frame, left, right):
        """Takes the points of the two lanes and creates a polyfit for each. 
        Visualization depents on the self.print_lanes param.

        Parameters
        ----------
        frame : array
            The initial frame
        left : list
            Points representing the left lane.
        right : list
            Points representing the right lane.

        Returns
        -------
        frame : array
            The frame with the two lanes vizualized, if self.print_lanes is True.
        array
            Left polyfit.
        array
            Right polyfit.
        """

        left_coef = self.fit_polyfit(left)
        right_coef = self.fit_polyfit(right)

        if self.print_lanes:
            self.visualize_lane(left_coef, frame, bgr_colour=(255, 128, 0))
            self.visualize_lane(right_coef, frame, bgr_colour=(0, 128, 255))

        return left_coef, right_coef

    def lanes_post_processing(self, frame, left_coef, left, right_coef, right, allowed_difference=None):
        """Finds the certainty for each lane.
        If the difference is greater than the allowed then delete the "uncertain" lane.

        Parameters
        ----------
        frame : array
            Image input
        left_coef : array
            Coefficients of left lane
        left : list
            Points representing the left lane
        right_coef : array
            Coefficients of right lane
        right : list
            Points representing the right lane
        allowed_difference : float
            percentege difference between the certainties of the two lanes. [0-100]

        Returns
        -------
        frame : array
            Image after printing the 2 cerainties
        left_coef : array | None
            of the left lane
        right_coef : array | None
            of the right lane
        trust_left_lane: boolean
            Indicating whether the left lane is trustworthy or not
        trust_right_lane: boolean
            Indicating whether the right lane is trustworthy or not
        """

        if allowed_difference is None:
            allowed_difference = self.allowed_certainty_perc_dif

        l_perc = self.find_lane_certainty(left_coef, self.prev_left_lane_coeffs, left)
        r_perc = self.find_lane_certainty(right_coef, self.prev_right_lane_coeffs, right)

        if self.print_lane_certainty:
            self.visualize_lane_certainty(frame, l_perc, r_perc)

        self.prev_left_lane_coeffs = left_coef
        self.prev_right_lane_coeffs = right_coef

        trust_l, trust_r = self.check_lane_certainties(
            l_perc, left_coef, r_perc, right_coef, frame, allowed_difference
        )

        trust_lk = self.trust_lane_keeping(l_perc, r_perc)

        return left_coef, right_coef, l_perc, r_perc, trust_l, trust_r, trust_lk

    # ------------------------------------------------------------------------------ #

    def find_lane_peaks(
        self,
        slice,
        height_norm,
        min_height,
        min_height_dif,
        pix_dif,
        allowed_peaks_width_error,
    ):
        """Finds 'square pulses' (inside the slice), coresponding to lane points.
        The width of a pulse depends on the height of the slice (height_norm).

        Parameters
        ----------
        slice : list
            list of a histogram that we want to find the lane points (square pulses).
        height_norm : float
            normalization of the height(bot=0,top=1) helps determine the width thressholds of a pulse.
        min_height : int
            the minimum height that a pulse must surpasses.
        pix_dif : int
            pixel number that helps the following param.
        min_height_dif :int
            the minimum difference that slice[i] and slice[i-pix_dif] must have in order to characterize it as 'square pulses'.
        allowed_peaks_width_error :int
            it's added to the peaks_max_width and it determines the upper threshold of the width of a 'square pulse'.

        Returns
        -------
        peaks : array
            all the detected lane points of the input slice.
        """

        peaks_max_width = self.peaks_max_width - (
            (self.peaks_max_width - self.peaks_min_width) * height_norm
        )
        uper_limit = peaks_max_width + allowed_peaks_width_error

        inside_a_peak = False
        # search for peaks
        peaks = []
        pix_num = 0
        height_dif_start = 0
        height_dif_end = 0
        for i in range(pix_dif, len(slice) - pix_dif):
            pixel = slice[i]

            height_dif_start = int(pixel) - int(slice[i - pix_dif])
            height_dif_end = int(pixel) - int(slice[i + pix_dif])

            if inside_a_peak:
                # peak finished
                if height_dif_end > min_height_dif:
                    inside_a_peak = False
                    # check if it's a lane peak
                    # temp = pix_num + pix_dif // 2
                    if pix_num >= self.peaks_min_width and pix_num <= uper_limit:
                        peak = i - (pix_num - pix_dif) // 2
                        peaks.append(peak)
                    pix_num = 0

                # still inside
                else:
                    if pixel > min_height:
                        pix_num += 1
                    else:
                        inside_a_peak = False

            # not inside a peak
            else:
                # go inside
                if pixel > min_height and height_dif_start > min_height_dif:
                    inside_a_peak = True
                    pix_num += 1
                # else stay outside

        return peaks

    def peaks_clustering(self, points, height, lanes):
        """Segments the points=(peaks[i],height) into lanes.
        1) Finds the best matchups between the detected points and the lanes.
        2) Appends the points that have a matchup into the corresponding lane.
        3) Insert the remaining points (the ones that do not have a matchup) [new lanes].

        Parameters
        ----------
        points : list
            containing the detected points of the slice.
        height : int
            the height of the slice. The y value of the peaks.
        lanes : list
            list of lists representing all the different lanes. Each lane (nested list), contains a set of points = [peaks[i],height].

        Returns
        -------
        lanes : list
            Updated list containing the detected points.
        """
        # INITIALIZE LANES AND EXITS
        if not lanes:
            return [[[x, height]] for x in points]

        lane_length = len(lanes)

        # For each lane we store the index of the best qualified point (int) and the distance (float).
        lanes_dict = [{"point_index": -1, "distance": -1} for _ in range(lane_length)]
        # For each point we store respectively if it's a qualified point for a lane (bool) and the index of this lane (int).
        points_dict = [{"used": False, "lane_index": -1} for _ in range(len(points))]

        # 1) Finds the best matchups between the detected points and the lanes.
        run_again = self.find_best_qualified_points(
            lanes_dict,
            points_dict,
            points,
            lanes,
            height,
        )
        # if some data were corrupted on the first run => run again
        if run_again:
            run_again = self.find_best_qualified_points(
                lanes_dict,
                points_dict,
                points,
                lanes,
                height,
            )

        # 2) Appends the points that have a matchup into the corresponding lane.
        # After finding the best (unique) point for each lane (if there is), append it and delete it from the list
        cnt = 0
        appended_cnt = 0
        for point in points_dict:

            if point["used"]:
                lanes[point["lane_index"]].append([points[cnt - appended_cnt], height])
                if self.custom_find_peaks:
                    points.pop(cnt - appended_cnt)
                else:
                    points = np.delete(points, cnt - appended_cnt)
                appended_cnt += 1
            cnt += 1

        # 3) Insert the remaining points (that do not have matchup) [new lanes].
        lanes_inserted = 0
        lanes_index = 0
        for point in points:
            # insert at the beginning
            if point < lanes[lanes_inserted][-1][0]:
                lanes.insert(lanes_inserted, [[point, height]])
                lanes_inserted += 1
                lanes_index = lanes_inserted

            # insert at the end
            elif point > lanes[lane_length + lanes_inserted - 1][-1][0]:
                lanes.append([[point, height]])
                lanes_inserted += 1

            # insert somewhere in the middle
            else:
                # find where we should insert it
                for j in range(lanes_index, lane_length + lanes_inserted - 1):
                    if point > lanes[j][-1][0] and point < lanes[j + 1][-1][0]:
                        lanes.insert(j + 1, [[point, height]])
                        lanes_inserted += 1
                        lanes_index = j + 1
                        break

        return lanes

    def find_best_qualified_points(
        self,
        lanes_dict,
        points_dict,
        detected_points,
        lanes,
        height,
    ):
        """
        Calculates the best matchup between the detected points and the existing lanes 
        in order to append them into those lanes. The information is past with the help
        of 2 dictionaries (lanes_dict, points_dict) so we update them accordingly (call
        by reference so no need to return). For each lane we check every peak inorder to
        find the best qualified.

        Parameters
        ----------
        lane_dict : dictionary
            For each lane we store the index of the best qualified point (int) and the 
            distance (float). [smaller distance => better point]
        points_dict : dictionary
            For each point we store respectively if it's a qualified point for a lane 
            (bool) and the index of this lane (int).
        detected_points : list
            contains the x values of the detected points
        lanes : list
            list of lists representing all the different lanes. Each lane (nested list),
            contains a set of points = [peaks[i],height].
        height : int
            the y value of all the detected points

        Returns
        -------
        run_again : bool
            returns True if some data were corrupted while updating to the best qualified
            point so we must run this function again, else False.
        """

        run_again = False

        for lane_index in range(len(lanes)):

            at_least_one = False

            # if we already found a point for this lane => means that this function is running for the second time => no need to check for this lane
            if lanes_dict[lane_index]["point_index"] != -1:
                continue

            # for every lane find the best qualified point
            for peak_index in range(len(detected_points)):

                x0, y0 = detected_points[peak_index], height
                x1, y1 = lanes[lane_index][-1][0], lanes[lane_index][-1][1]
                width_dist = abs((x0 - x1) * self.step / (y0 - y1))

                # Check only for near points
                if width_dist < self.max_allowed_dist:

                    flag, width_dist_error = self.verify_with_expected_value(
                        lanes[lane_index], height, detected_points[peak_index]
                    )

                    if flag:
                        at_least_one = True
                        temp_dist = (
                            self.weight_for_width_distance * width_dist
                            + self.weight_for_expected_value_distance * width_dist_error
                        )

                        # if point not used by any lane and lane has not been initialized with best qualified point
                        if (
                            not points_dict[peak_index]["used"]
                            and lanes_dict[lane_index]["point_index"] == -1
                        ):

                            self.add_qualified_point(
                                lanes_dict,
                                points_dict,
                                lane_index,
                                peak_index,
                                temp_dist,
                            )

                        # if point used by other lane and lane has not a qualified point (1 point 2 lane) => Check which lane is the best fit for this lane
                        elif (
                            points_dict[peak_index]["used"] and lanes_dict[lane_index]["point_index"] == -1
                        ):

                            # get prev lane
                            prev_lane_index = points_dict[peak_index]["lane_index"]

                            # if distance from prev lane is greater than this lane
                            if lanes_dict[prev_lane_index]["distance"] > temp_dist:
                                run_again = True

                                # delete point from prev lane
                                lanes_dict[prev_lane_index]["point_index"] = -1
                                lanes_dict[prev_lane_index]["distance"] = -1
                                # For this lane : initialize with the point, For point : update only the used lane (it's already used)
                                self.add_qualified_point(
                                    lanes_dict,
                                    points_dict,
                                    lane_index,
                                    peak_index,
                                    temp_dist,
                                )

                        # if point not used by any lane and lane has already a qualified point (2 points 1 lane) => Check which point is the best
                        elif (
                            not points_dict[peak_index]["used"]
                            and lanes_dict[lane_index]["point_index"] != -1
                        ):

                            # get prev point
                            prev_point_index = lanes_dict[lane_index]["point_index"]

                            # for the same lane, if the already qualified point has greater distance than this point => New point is the best
                            if lanes_dict[lane_index]["distance"] > temp_dist:
                                run_again = True

                                # delete data from prev point (not used and lane_index)
                                points_dict[prev_point_index]["used"] = False
                                points_dict[prev_point_index]["lane_index"] = -1
                                # Update lane data (new point_index and new distance) and Initialize data for new point
                                self.add_qualified_point(
                                    lanes_dict,
                                    points_dict,
                                    lane_index,
                                    peak_index,
                                    temp_dist,
                                )

                        # if point is used by another lane and lane has already another best fit (2 points 2 lanes) => Check if the new combination is better
                        elif (
                            points_dict[peak_index]["used"] and lanes_dict[lane_index]["point_index"] != -1
                        ):

                            # 2 cases :
                            # 1) this combination is already stored (exit)
                            # 2) else (check)
                            if (
                                points_dict[peak_index]["lane_index"] == lane_index
                                and lanes_dict[lane_index]["point_index"] == peak_index
                            ):
                                continue

                            prev_lane_index = points_dict[peak_index]["lane_index"]
                            prev_point_index = lanes_dict[lane_index]["point_index"]

                            # if new distance is smaller than the 2 others (delete data from prev point, delete data from prev lane,update data for this point and this lane)
                            if (
                                lanes_dict[lane_index]["distance"] > temp_dist
                                and lanes_dict[prev_lane_index]["distance"] > temp_dist
                            ):
                                run_again = True

                                # delete data from prev point (not used and lane_index)
                                points_dict[prev_point_index]["used"] = False
                                points_dict[prev_point_index]["lane_index"] = -1
                                # delete point from prev lane
                                lanes_dict[prev_lane_index]["point_index"] = -1
                                lanes_dict[prev_lane_index]["distance"] = -1
                                # Update lane data (new point_index and new distance) and update data for new point
                                self.add_qualified_point(
                                    lanes_dict,
                                    points_dict,
                                    lane_index,
                                    peak_index,
                                    temp_dist,
                                )

                elif at_least_one:
                    break

        # Check if we realy need to run again
        if run_again:
            all_points = True
            for i in points_dict:
                if not i["used"]:
                    all_points = False
                    break
            all_lanes = True
            for i in lanes_dict:
                if i["point_index"] == -1:
                    all_lanes = False
                    break

            if all_lanes or all_points:
                run_again = False

        return run_again

    def add_qualified_point(self, lanes_dict, points_dict, lane_index, peak_index, distance):
        """Adds the input data (lane_index, peak_index, distance), into the input 
        dictionaries (lanes_dict, points_dict). We run this function when we have
        a new qualified point.

        Parameters
        ----------
        lane_dict : dictionary
            Here we store for every lane the index of the best qualified point (int) 
            and the distance (float). [smaller distance => better point]
        points_dict : dictionary
            For every point we store respectively if it's a qualified point for a 
            lane (bool) and the index of this lane (int).
        lane_index : int
            The index of the lane
        peak_index : int
            The index of the qualified point
        distance : float
            The distance of the point and the lane.
        """

        lanes_dict[lane_index]["point_index"] = peak_index
        lanes_dict[lane_index]["distance"] = distance
        points_dict[peak_index]["used"] = True
        points_dict[peak_index]["lane_index"] = lane_index

    def verify_with_expected_value(self, lane, height, x_value):
        """
        This function verifies if the input point [height, x_value] can be added
        on the input lane by :
        Calculating some expected values (the number depends on the len(lane) value)
        of the inputed lane, in the inputed height. The distances are the expected
        values subtracted by the x_value, and it only cares about the smallest one 
        (distance). Calculates a punishment depending on the len(lane) and the 
        difference in height, between the last point of the lane and the currenct 
        height. Finally, it returns whether we should add this point (bool) (if 
        distance is smaller than what's allowed) and the distance with the added 
        punishment (float).

        Parameters
        ----------
        lane : list
            a list containing all the points [x,y] of this lane.
        height : int
            the y value of the point that we found
        x_value : int
            the x value of the point that we want to add.

        Returns
        -------
        True|False : bool
            whether we can add this point=[height,x_value] in this lane or not.
        distances : float
            the calculated distance with the added punishment.
        """

        distances = []

        # create a lane from the first and last lane points, then calculate the expected peak (when y=height)

        # We will calculate the expected peak = x0.
        x_dif, y_dif = lane[-1][0] - lane[0][0], lane[-1][1] - lane[0][1]
        x_div_y = x_dif / y_dif if y_dif != 0 else 0

        # x/y = (x0 - x1) / (y0 - y1)               we solve for x0 = expected_peak
        # x/y = x_div_y , x0 = EXPECTED,  x1 = lane[-1][0], y0 = height, y1 = lane[-1][1]
        expected_peak_1 = (height - lane[-1][1]) * x_div_y + lane[-1][0]
        distances.append(abs(x_value - expected_peak_1))

        # punishment if we only have one point
        punish = self.max_allowed_dist // 2 if x_div_y == 0 else 0
        # punishment depending on the count of the slices that preceded from the last point of the lane
        perc_of_preceded_slices = ((height - lane[-1][1]) / self.step - 1) / self.real_slices
        punish += perc_of_preceded_slices * self.max_allowed_dist

        if len(lane) > 3:
            # create a lane from the last 2 lane points and calculate the expected peak.
            x_dif_2, y_dif_2 = lane[-1][0] - lane[-2][0], lane[-1][1] - lane[-2][1]
            x_div_y_2 = x_dif_2 / y_dif_2 if y_dif_2 != 0 else 0
            expected_peak_2 = (height - lane[-1][1]) * x_div_y_2 + lane[-1][0]
            distances.append(abs(x_value - expected_peak_2))

            # same for the second-to-last and third-to-last points.
            x_dif_3, y_dif_3 = lane[-2][0] - lane[-3][0], lane[-2][1] - lane[-3][1]
            x_div_y_3 = x_dif_3 / y_dif_3 if y_dif_3 != 0 else 0
            expected_peak_3 = (height - lane[-2][1]) * x_div_y_3 + lane[-2][0]
            distances.append(abs(x_value - expected_peak_3))

        dist = min(distances)
        max_allowed_dist = self.max_allowed_dist if x_div_y == 0 else self.max_allowed_dist // 2

        return (True, dist + punish) if dist < max_allowed_dist else (False, dist + punish)

    # ------------------------------------------------------------------------------ #

    def calculate_lane_boundaries(self, lane_points):
        """
        Calculates the points of the first and last slice based on the line
        forming by the first and last identified point of the inputted lane.

        Parameters
        ----------
        lane_points : array
            points representing a lane.

        Returns
        -------
        top : float
            the width value at the top (The x value of the calculated line when y=top_height).
        bot : float
            the width value at the bottom (The x value of the calculated line when y=bot_height).
        """

        x = [lane_points[0][0], lane_points[-1][0]]
        y = [lane_points[0][1], lane_points[-1][1]]

        # x = Slope * y + b
        # x = Width, y =  Height
        slope = (x[1] - x[0]) / (y[1] - y[0])
        b = x[0] - slope * y[0]

        top = slope * self.top_row_index + b
        bot = slope * self.bottom_row_index + b

        return top, bot

    # ------------------------------------------------------------------------------ #

    def fit_polyfit(self, lane, percentage_for_first_degree=0.3):
        """Converts list to a polynomial. The degree depends on the percentage
        of the detected points.

        Parameters
        ----------
        lane : list
            Points representing a lane.
        percentage_for_first_degree : float
            Determines the degree of the polynomial (first or second), it's a float number [0-1].

        Returns
        -------
        lane_coef : array
            the coefficients of the polynomial, len(lane_coef) is 3.
        """
        lane_coef = None

        # if there is a lane
        if len(lane) > 0:
            lane_y_x = [[peak[1], peak[0]] for peak in lane]

            if len(lane_y_x) > self.slices * percentage_for_first_degree:
                lane_coef = self.fit_polynomial_with_degree(lane_y_x, degree=2)
                lane_coef = self.check_for_extreme_coefs(lane_coef)
            else:
                lane_coef = self.fit_polynomial_with_degree(lane_y_x, degree=1)
                lane_coef = self.check_for_extreme_coefs(lane_coef)
                if lane_coef is not None:
                    lane_coef = np.array([0, lane_coef[0], lane_coef[1]])

        return lane_coef

    def fit_polynomial_with_degree(self, points, degree=2):
        """Converts points to a polynomial with the imported degree.

        Parameters
        ----------
        points : list
            the set of points that will create the polynomial.
        degree : int
            the degree of the polynomial. (1: first, 2: second degree ...)

        Returns
        -------
        coef : array
            the coefficients of the polynomial. len(coef) = degree
        """
        # Extract the x and y coordinates of the points
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        # Create the design matrix
        matrix = np.vander(x, degree + 1)

        # Solve the least squares problem to find the coefficients
        coef, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)

        return coef

    def check_for_extreme_coefs(self, coefs):
        """Checks and deletes extreme coefs

        Parameters
        ----------
        coefs : array
            of a lane.

        Returns
        -------
        array | None
            the coefs or None if there is an extreme value.

        """

        limits = {3: self.extreme_coef_second_deg, 2: self.extreme_coef_first_deg}
        return coefs if (coefs is not None and math.fabs(coefs[0]) < limits[len(coefs)]) else None

    def visualize_lane(self, coefs, frame, bgr_colour=(102, 0, 102)):
        """Simple visualize a lane with the imported coefs to the imported frame.

        Parameters
        ----------
        coefs : array
            of a lane
        frame : array
            Input image
        bgr_colour : tuple
            colour for the visualization of the lane

        Results
        -------
        frame : array
            the image with the lane if self.print_lanes is True.
        """
        # check if we can or want the visualization to happen
        if coefs is None:
            return

        a, b, c = coefs[0], coefs[1], coefs[2]

        end = int((1 - self.bottom_perc) * self.height)
        for i in range(self.height, end, -3):
            start = (int(a * i**2 + b * i + c), i)
            k = i - 3
            end = (int(a * k**2 + b * k + c), k)

            # No need but for rebustness
            if not (start[0] < 0 or start[0] > self.width or end[0] < 0 or end[0] > self.width):
                cv2.line(frame, start, end, bgr_colour, thickness=3)

        return

    # ------------------------------------------------------------------------------ #

    def find_lane_certainty(self, new_coeffs, prev_coeffs, peaks):
        """Finds the certainty of a lane and returns a percentage generated equaly by the:
        1) Similarity with the previous lane and the
        2) Length of the first and last peak.

        Parameters
        ----------
        new_coeffs : array
            of the new detected lane
        prev_coeffs : array
            of the lane that was detected on the previous frame
        peaks : list
            Points representing the lane

        Returns
        -------
        certainty : float
            a percentage of the certainty of the imported lane.

        """

        if prev_coeffs is None or new_coeffs is None:
            return 0.0

        # Similarity
        error = np.sqrt(np.mean((new_coeffs - prev_coeffs) ** 2))
        similarity = 100 - error
        if similarity < 0:
            similarity = 0

        # Lane Peaks
        # 1)
        # peaks_percentage = len(peaks) / self.slices * 100
        # 2)
        peaks_percentage = (peaks[-1][1] - peaks[0][1]) / (self.top_row_index - self.bottom_row_index) * 100

        certainty = (
            self.certainty_perc_from_peaks * peaks_percentage
            + (1 - self.certainty_perc_from_peaks) * similarity
        )

        return round(certainty, 2)

    def trust_lane_keeping(self, l_certainty, r_certainty):
        """Checks if we can trust the lane detection

        Parameters
        ----------
        l_certainty : float
            a percentage of the certainty of the left lane.
        r_certainty : float
            a percentage of the certainty of the right lane.

        Returns
        -------
        bool
            if we can trust the lane keeping with the detected lanes
        """

        # Is lane keeping trusted?
        single_flag = (
            l_certainty > self.min_single_lane_certainty or r_certainty > self.min_single_lane_certainty
        )
        dual_flag = (
            l_certainty > self.min_dual_lane_certainty and r_certainty > self.min_dual_lane_certainty
        )
        lane_keeping_ok = single_flag or dual_flag

        # Return this
        result = self.prev_trust_lk and lane_keeping_ok
        # Update previous trust
        self.prev_trust_lk = True if lane_keeping_ok else False

        return result

    def check_lane_certainties(
        self,
        left_certainty,
        left_lane_coeffs,
        right_certainty,
        right_lane_coeffs,
        frame,
        allowed_difference=None,
    ):
        """This function compares the certainties of the left and right lanes and returns flags
        indicating whether they are trustworthy or not. If the difference in certainty between
        the lanes is greater than a specified threshold, the function sets the corresponding
        flag to False, indicating that the lane should not be trusted. Otherwise, the flag is
        set to True, indicating that the lane is reliable. If the function determines that a
        lane is not trustworthy, it colors it gray in the corresponding frame.

        Parameters
        ----------
        left_certainty : float
            percentage corresponding to the certainty of the left lane
        left_lane_coeffs : array
           left lane polynomial
        right_certainty : float
            percentage corresponding to the certainty of the right lane
        right_lane_coeffs : array
           right lane polynomial
        frame : array
           Input image
        allowed_defference : float
           percentege difference between the certainties of the two lanes. [0-100]

        Returns
        -------
        bool
            trust_left_lane
        bool
            trust_right_lane
        """
        if allowed_difference is None:
            allowed_difference = self.allowed_certainty_perc_dif

        if np.abs(left_certainty - right_certainty) > allowed_difference:
            if left_certainty > right_certainty:
                if self.print_lanes:
                    self.visualize_lane(right_lane_coeffs, frame, (169, 169, 169))
                return True, False
            else:
                if self.print_lanes:
                    self.visualize_lane(left_lane_coeffs, frame, (169, 169, 169))
                return False, True

        return True, True

    # ------------------------------------------------------------------------------ #

    def visualize_all_peaks(self, frame, peaks, bgr_colour=(255, 0, 255)):
        """Takes a frame and a list of peaks as input, and returns an image 
        with circles drawn around each peak.

        Parameters
        ----------
        frame : array
            Input image
        peaks : list
            of lists [x,y] representing the peaks in the frame.
        bgr_colour : tuple
            An optional parameter representing the color of the circles.
            The default value is (255, 0, 255), which is a shade of pink.

        Returns
        -------
        frame : array
            The image frame with circles drawn around each peak
        """

        for peak in peaks:
            point = (peak[0], peak[1])
            cv2.circle(frame, point, 2, bgr_colour, 2)

        return

    def visualize_peaks(self, frame, left, right):
        """This function takes a frame and 2 lists (representing the points of
        each lane) as input, and returns an image with circles drawn around each point.

        Parameters
        ----------
        frame : array
            Input image
        left : list
            points of the left lane
        right : list
            points of the right lane

        Returns
        -------
        frame : array
            The image frame with circles drawn around each point
        """

        # visualize the Points
        for peak in left:
            point = (peak[0], peak[1])
            cv2.circle(frame, point, 2, (0, 0, 0), 2)

        for peak in right:
            point = (peak[0], peak[1])
            cv2.circle(frame, point, 2, (0, 0, 0), 2)

        return

    def visualize_lane_certainty(self, frame, l_perc, r_perc):
        """This function takes a frame, the 2 lane certainty percentages and visualize them

        Parameters
        ----------
        frame : array
            Input image
        l_perc : float
            certainty of the left lane
        r_perc : float
            certainty of the right lane
        """
        cv2.putText(
            frame,
            f"{l_perc}",
            (int(0.2 * self.width), int(self.height / 2.5)),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 128, 0),
            1,
        )
        cv2.putText(
            frame,
            f"{r_perc}",
            (int(0.7 * self.width), int(self.height / 2.5)),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 128, 255),
            1,
        )

        return

    # ------------------------------------------------------------------------------ #
    def is_dashed(self, lane):

        max_allowed_spaces_inside_a_dashed_lane = 1
        lane_index = 0
        inside_dashed_line = True if self.bottom_row_index == lane[0][1] else False
        point_cnt = 0
        start = True
        last_spaces_in_a_row_cnt = 0
        change_state_count = 0
        length = len(lane)
        for height in range(
            self.bottom_row_index,
            self.top_row_index - 1,
            self.step,
        ):

            # We have a point in this height

            if height == lane[lane_index][1]:

                if inside_dashed_line:
                    # TODO: check if we really need this
                    if last_spaces_in_a_row_cnt > 0:
                        point_cnt += last_spaces_in_a_row_cnt
                    point_cnt += 1
                else:
                    # if between the margins then we had a space (space = area between the dashed lanes)
                    # spaces do not have (false detected) points
                    if (point_cnt < self.max_points and point_cnt > self.min_points_space) or start:
                        start = False
                        inside_dashed_line = True
                        change_state_count += 1
                        point_cnt = 1
                    else:
                        # No need to check further
                        if change_state_count >= self.min_count_of_dashed_lanes:
                            return True
                        return False

                last_spaces_in_a_row_cnt = 0

                lane_index += 1
                if length <= lane_index:
                    break

            # We do not have a point in this height
            elif height > lane[lane_index][1]:

                last_spaces_in_a_row_cnt += 1

                if not inside_dashed_line:
                    point_cnt += 1
                else:
                    if last_spaces_in_a_row_cnt <= max_allowed_spaces_inside_a_dashed_lane:
                        pass
                    else:
                        # if between the margins then we had a space (space = area between the dashed lanes)
                        # spaces do not have (false detected) points
                        if (point_cnt < self.max_points and point_cnt > self.min_points) or start:
                            start = False
                            inside_dashed_line = False
                            change_state_count += 1
                            point_cnt = last_spaces_in_a_row_cnt
                        else:
                            # No need to check further
                            if change_state_count >= self.min_count_of_dashed_lanes:
                                return True

                            return False

            else:
                self.log.info(">>>>>> ERROR: height of lane point is smaller.")
                self.log.info(lane)
                break

        if change_state_count >= self.min_count_of_dashed_lanes:
            return True
        return False

    def check_for_dashed(self, left, right, frame=None):

        dashed_l, dashed_r = False, False
        if left:
            dashed_l = self.is_dashed(left)
            if dashed_l and self.print_if_dashed and frame is not None:
                cv2.putText(
                    frame,
                    "Dashed",
                    (int(0.2 * self.width), int(self.height / 3.5)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 128, 0),
                    1,
                )

        if right:
            dashed_r = self.is_dashed(right)
            if dashed_r and self.print_if_dashed and frame is not None:
                cv2.putText(
                    frame,
                    "Dashed",
                    (int(0.7 * self.width), int(self.height / 3.5)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 128, 255),
                    1,
                )
        return dashed_l, dashed_r

    # ------------------------------------------------------------------------------ #

    def choose_405(self):
        self.is_horizontal = False

        self.bottom_perc = float(self.config["LANE_DETECT"].get("bottom_perc_405"))
        self.peaks_min_width = int(self.config["LANE_DETECT"].get("peaks_min_width_405"))
        self.peaks_max_width = int(self.config["LANE_DETECT"].get("peaks_max_width_405"))

        self.bottom_row_index = self.height - self.bottom_offset
        end = int((1 - self.bottom_perc) * self.height)
        self.step = int(-(self.height * self.bottom_perc / self.slices))
        self.real_slices = int((end - self.bottom_row_index) // self.step)
        self.top_row_index = self.bottom_row_index + self.real_slices * self.step

        self.height_norm = np.linspace(0, 1, self.real_slices + 1)

        # For choosing lanes (Half image) (check if lanes difference is bigger/smaller than these params)
        self.bottom_width = int(self.config["LANE_KEEPING"].get("bottom_width_405"))
        self.top_width = int(self.config["LANE_KEEPING"].get("top_width_405"))

        self.min_top_width_dif = self.top_width * self.min_lane_dist_perc
        self.max_top_width_dif = self.top_width * self.max_lane_dist_perc
        self.min_bot_width_dif = self.bottom_width * self.min_lane_dist_perc
        self.max_bot_width_dif = self.bottom_width * self.max_lane_dist_perc

        max_dash_points_perc = float(self.config["LANE_DETECT"].get("dashed_max_dash_points_perc"))
        min_dash_points_perc = float(self.config["LANE_DETECT"].get("dashed_min_dash_points_perc"))
        min_space_points_perc = float(self.config["LANE_DETECT"].get("dashed_min_space_points_perc"))
        self.min_count_of_dashed_lanes = float(self.config["LANE_DETECT"].get("dashed_min_count_of_dashed_lanes"))

        # Checks if len of dashed lanes and len of spaces are between the following margins
        self.max_points = max_dash_points_perc * self.real_slices
        self.min_points = min_dash_points_perc * self.real_slices
        self.min_points_space = min_space_points_perc * self.real_slices

    def choose_455(self):
        self.is_horizontal = False

        self.bottom_perc = float(self.config["LANE_DETECT"].get("bottom_perc_455"))
        self.peaks_min_width = int(self.config["LANE_DETECT"].get("peaks_min_width_455"))
        self.peaks_max_width = int(self.config["LANE_DETECT"].get("peaks_max_width_455"))

        self.bottom_row_index = self.height - self.bottom_offset
        end = int((1 - self.bottom_perc) * self.height)
        self.step = int(-(self.height * self.bottom_perc / self.slices))
        self.real_slices = int((end - self.bottom_row_index) // self.step)
        self.top_row_index = self.bottom_row_index + self.real_slices * self.step

        self.height_norm = np.linspace(0, 1, self.real_slices + 1)

        # For choosing lanes (Half image) (check if lanes difference is bigger/smaller than these params)
        self.bottom_width = int(self.config["LANE_KEEPING"].get("bottom_width_455"))
        self.top_width = int(self.config["LANE_KEEPING"].get("top_width_455"))

        self.min_top_width_dif = self.top_width * self.min_lane_dist_perc
        self.max_top_width_dif = self.top_width * self.max_lane_dist_perc
        self.min_bot_width_dif = self.bottom_width * self.min_lane_dist_perc
        self.max_bot_width_dif = self.bottom_width * self.max_lane_dist_perc

        max_dash_points_perc = float(self.config["LANE_DETECT"].get("dashed_max_dash_points_perc"))
        min_dash_points_perc = float(self.config["LANE_DETECT"].get("dashed_min_dash_points_perc"))
        min_space_points_perc = float(self.config["LANE_DETECT"].get("dashed_min_space_points_perc"))
        self.min_count_of_dashed_lanes = float(self.config["LANE_DETECT"].get("dashed_min_count_of_dashed_lanes"))

        # Checks if len of dashed lanes and len of spaces are between the following margins
        self.max_points = max_dash_points_perc * self.real_slices
        self.min_points = min_dash_points_perc * self.real_slices
        self.min_points_space = min_space_points_perc * self.real_slices

    def horizontal_line(self):
        self.choose_455()

        self.is_horizontal = True
        perc_from_mid = float(self.config["LANE_DETECT"].get("hor_perc_from_mid"))
        self.bottom_row_index = int((self.height // 2) + self.height * perc_from_mid)
        end = int((self.height // 2) - self.height * perc_from_mid)
        self.step = int(-(self.height * 2 * perc_from_mid / self.slices))
        self.real_slices = int((end - self.bottom_row_index) // self.step)
        self.top_row_index = int(self.bottom_row_index + self.real_slices * self.step)
        self.print_lanes = True
        self.print_peaks = True
        self.peaks_max_width = int(self.config["LANE_DETECT"].get("hor_peaks_max_width"))
        self.peaks_min_width = int(self.config["LANE_DETECT"].get("hor_peaks_min_width"))
        self.square_pulses_allowed_peaks_width_error = int(
            self.config["LANE_DETECT"].get("hor_square_pulses_allowed_peaks_width_error")
        )
