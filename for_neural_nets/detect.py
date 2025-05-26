import configparser
import math

import cv2
import numpy as np
# from scipy.signal import find_peaks


class LaneDetection:
    """
    This class is a copy of the lane detection module "/src/LaneDetection/detection.py", 
    modified to run only peaks_detection, so it can be used in the labeling process /for_neural_nets/labeling.py.  
    """

    def __init__(self, width, height, camera, lk):

        self.height = height
        self.width = width
        self.lk = lk

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
        self.minimum = self.square_pulses_min_height
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

    def lanes_detection(self, src):
        lanes, peaks, gray = self.peaks_detection(src)
        return lanes,peaks 

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

            # TODO : Make an RNN for finding the peaks
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

        return lanes, peaks, src

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
                        peak = i - (pix_num// 2)
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
                    pix_num = 1
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

