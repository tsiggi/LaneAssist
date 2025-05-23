from data_helper import save_annotated_data
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
from detect import LaneDetection


class LanePeakLabeler:
    def __init__(self, video_path=None, image_path=None, output_dir = "lane_anotation_dataset"):
        """
        Initialize the lane peak labeler tool
        
        Args:
            video_path: Path to video file (optional)
            image_path: Path to image file (optional)
            output_dir: Directory to save labeled data
        """
        self.video_path = video_path
        self.image_path = image_path
        self.output_dir = output_dir
        
        # Parameters
        self.max_lanes = 5 
        self.bottom_offset = 50 # in pixels
        self.bottom_perc = 0.5 # percentage of the height
        self.slices = 80        # number of slices
        self.colours = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
            (128, 0, 128), (0, 128, 128), (128, 128, 0)
        ]
        self.skipped_frames = 10 # Number of frames to skip in video
        self.skip_first_frames = 0
        self.height_minimum_difference = 2 # Minimum height difference to consider a peak
        
        # Initialize data structures
        self.histograms_peaks = {}  # Key: slice height, Value: list of widths 
        self.LD_peaks = {}
        self.lanes = None # store lists of lane points depending on the self.max_lanes 

        # Helpful variables
        self.cap = None # VideoCapture object
        self.clicks = []
        self.flag_go_back = False
        self.flag_skip_this_frame = False
        self.flag_click_is_the_peak = False
        self.LD = None

    def start_labeling(self):
        # load the source [image or frame of a video (repeat for every frame)]
        while self.load_source(): 
            image_for_saving = self.src.copy() 
            frame = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
            self.compute_heights_of_histogram(frame.shape[0])

            self.find_peaks_using_lanedetection()
            if self.flag_skip_this_frame:
                self.flag_skip_this_frame = False
                continue

            # In a while loop cause we need to change self.height when the user goes back (to the previous slice)
            # bottom > top --->  step < 0
            self.height = self.bottom_row_index
            while self.height > self.top_row_index - 1:
                self.clicks = [] # reset the clicks for each slice
                if self.height in self.LD_peaks.keys():
                    self.histograms_peaks[self.height] = self.LD_peaks[self.height] 
                else : 
                    self.histograms_peaks[self.height] = [] 
                # self.histograms_peaks[self.height] = [] # init for this slice
                
                self.visualized_frame = self.src.copy()
                self.histogram = [int(x) for x in frame[self.height]]

                self.print_img_and_histogram(self.visualized_frame, self.height)

                if self.flag_skip_this_frame:
                    break

                # if flag is checked go to the previous slice 
                if not self.flag_go_back : 
                    self.visualize_hist_peaks()
                    self.histograms_peaks[self.height].sort()
                else :
                    self.height = self.height - self.step if self.height < self.bottom_row_index else self.bottom_row_index
                    self.flag_go_back = False
                    continue

                self.height += self.step # add the negative step 
            
            if self.flag_skip_this_frame:
                self.flag_skip_this_frame = False
                continue

            # Classify points into lanes 
            self.lanes = self.classify_into_lanes()

            save_annotated_data(image_for_saving, self.histograms_peaks, self.lanes, output_dir=self.output_dir)

            self.visualize_lane_points(image_for_saving, self.lanes)

            self.reset_params() 
            cv2.destroyAllWindows()

    def find_peaks_using_lanedetection(self):
        """Phase 1: Find peaks using Ld and let user accept or discard them."""
        lanes, points = self.LD.lanes_detection(self.src.copy())
        
        img = self.src.copy() 

        def wait_for_key():
            while True:
                key = cv2.waitKey(0)
                if key == -1:
                    continue
                key_char = chr(key & 0xFF)  # convert to character
                if  key_char == 'y':
                    return False, True
                elif key_char == 'n':
                    return False, False
                elif key_char == 'q': 
                    self.flag_skip_this_frame = True
                    return True, False 
                elif key_char == '0':
                    print(">>> Continue withoud Auto-Detection")
                    return True, True 
                elif key_char == '!':
                    print(">>> Exiting...")
                    exit(1)
        for point in points : 
            cv2.circle(img, (point[0], point[1]), 5, (0,0,0), -1)

        for point in points: 
            printable_img = img.copy() 
            x, y = point[0], point[1] 
            cv2.circle(printable_img, (x, y), 18, (0, 0, 255), 4)
            cv2.imshow("LD CHECK", printable_img)
            exit, accept = wait_for_key()

            if exit : 
                # if accept : 
                    # cv2.destroyWindow("LD CHECK")
                return 
            if accept : 
                cv2.circle(img, (x, y), 6, (0,255,0), -1)
                if y in self.LD_peaks.keys() : 
                    self.LD_peaks[y].append(x)
                else :
                    self.LD_peaks[y] = [x] 
            else : 
                cv2.circle(img, (x, y), 6, (0,0,0), -1)
        cv2.destroyWindow("LD CHECK")
    
    def reset_params(self):
        self.histograms_peaks = {}
        self.lanes = None
        self.flag_go_back = False
        self.LD_peaks = {}

    def classify_into_lanes(self):
        """Phase 3: Classify detected points into lanes"""
        
        
        def wait_for_key():
            while True:
                key = cv2.waitKey(0)

                if key == -1:
                    continue

                key_char = chr(key & 0xFF)  # convert to character
                
                if key_char in ['1','2','3','4','5','6','7','8','9']:
                    current_lane = int(key_char) - 1 
                    cv2.circle(self.classification_image, (x, y), 6, self.colours[current_lane], -1)
                    cv2.circle(self.src, (x, y), 6, self.colours[current_lane], -1)
                    cv2.imshow("src", self.classification_image)
                    cv2.waitKey(1)
                    return True, current_lane 

                elif  key_char == 'n':
                    return True, None

                elif key_char == 'b':
                    return False, None
        
        peak_to_lane = [] # in each index [[height,width], lane_id]

        flag_previous_height = False
        height = self.bottom_row_index
        i = 0 
        while height > self.top_row_index - 1:
            points = self.histograms_peaks[height]
            
            while i < len(points):
                current_lane = None
                self.classification_image = self.src.copy()

                x, y = points[i], height
                cv2.circle(self.classification_image, (x, y), 18, (0, 0, 255), 4)
                cv2.imshow("src", self.classification_image)

                go_to_next, current_lane = wait_for_key()

                if go_to_next: 
                    peak_to_lane.append([(x, y), current_lane])
                    i += 1
                else : 
                    if len(peak_to_lane) > 0:
                        peak_to_lane.pop()
                    if i > 0:
                        i -= 1
                    else : 
                        flag_previous_height = True
                        break

            i = 0 
            if flag_previous_height: 
                flag_previous_height = False
                height = height - self.step 
                if height > self.bottom_row_index :
                    height = self.bottom_row_index
                i = len(self.histograms_peaks[height]) - 1
                flag_run = True 
                while i < 0 and flag_run: 
                    height = height - self.step 
                    if height > self.bottom_row_index: 
                        flag_run = False
                        height = self.bottom_row_index
                        i = 0
                    else : 
                        i = len(self.histograms_peaks[height]) - 1
            else : 
                height += self.step 

        # store points (that have as label a lane) into list of lanes 
        lanes = []
        for i in range(self.max_lanes):
            lanes.append([])   
        for peak in peak_to_lane:
            if peak[1] != None:
                lanes[peak[1]].append(peak[0])

        return lanes 

    def find_peak_from_user_clicks(self, left_x, y, right_x):
        i = left_x
        step = 2 if left_x < right_x else -2
        cnt = 0 

        while (self.histogram[i] - y < self.height_minimum_difference) and ((left_x <= i and i < right_x) or (right_x < i and i <= left_x)):
            i+= step
            cnt += 1
        if cnt < 1 or (i > right_x and step > 0) or (i < left_x and step > 0):
            return None
        x_1 = i - step / 2
        cnt = 0
        while (self.histogram[i] - y > self.height_minimum_difference) and ((left_x <= i and i < right_x) or (right_x < i and i <= left_x)):
            i+= step
            cnt += 1
        if cnt < 1 or (i > right_x and step > 0) or (i < left_x and step > 0):
            return None
        x_2 = i - step / 2
        return int((x_1 + x_2) / 2)

    def make_click_a_peak(self, event):
        """Make the click a peak"""
        if event.xdata is None or event.ydata is None:
            return
        
        x = int(event.xdata)
        y = int(event.ydata)

        # save the lane point 
        self.histograms_peaks[self.height].append(x)
    
        # Update visualization on both histogram and image
        self.visualize_point_on_image(x, self.height)

    def on_click(self, event):
        """Handle mouse clicks on the histogram plot"""
        if event.xdata is None or event.ydata is None:
            return
        
        if self.flag_click_is_the_peak: 
            self.make_click_a_peak(event)
            self.flag_click_is_the_peak = False
            return
        
        x = int(event.xdata) # width 
        y = int(event.ydata) # height
        
        # Save the click coordinates
        self.clicks.append((x, y))
        
        plt.plot(x, y, 'ko', markersize=2)
        plt.show()

        # After each pair of clicks, process them to find the peak center
        if len(self.clicks) % 2 == 0:
            left_x, left_y = self.clicks[-2]
            right_x, right_y = self.clicks[-1]
            

            # Find the the peak
            peak_x = self.find_peak_from_user_clicks(left_x, max(left_y,right_y), right_x)
            if peak_x is not None:
                # save the lane point 
                self.histograms_peaks[self.height].append(peak_x)
            
                # Update visualization on both histogram and image
                self.visualize_point_on_image(peak_x, self.height)

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'n' or event.key == ' ':  # Next slice
            plt.close()
        elif event.key == 'r':  # Previous slice
            if self.clicks:
                self.visualize_deleted_click(self.clicks[-1][0], self.clicks[-1][1])
                self.clicks.pop()
        elif event.key == 'd':  # Delete last peak
            if self.histograms_peaks[self.height]:
                peak_x = self.histograms_peaks[self.height].pop()
                self.visulize_deleted_peak(peak_x, self.height)
        elif event.key == 'b':  # Previous slice
            plt.close()
            self.flag_go_back = True
            
        elif event.key == 'q':  # Quit and save
            plt.close()
            self.flag_skip_this_frame = True
            cv2.destroyWindow("src")
        
        elif event.key == '!': # Terminate 
            print(">>> Exiting...")
            exit(1)
        
        elif event.key == 'c': # Click is the peak
            self.flag_click_is_the_peak = True

    def visualize_hist_peaks(self):
        for peaks in self.histograms_peaks[self.height]:
            peak_x = peaks
            cv2.circle(self.src, (peak_x, self.height), 3, (255, 0, 255), 3)
            cv2.imshow("src", self.src)
            cv2.waitKey(1)

    def visualize_point_on_image(self, peak_x, current_height, show_plt = True):
        cv2.circle(self.visualized_frame, (peak_x, current_height), 3, (230, 216, 173), 3)
        cv2.imshow("src", self.visualized_frame)
        cv2.waitKey(1)
        plt.plot(peak_x, self.histogram[peak_x], 'go', markersize=8)
        if show_plt :
            plt.show()  

    def visulize_deleted_peak(self, peak_x, current_height):
        cv2.circle(self.visualized_frame, (peak_x, current_height), 3, (0, 0, 0), 3)
        cv2.imshow("src", self.visualized_frame)
        cv2.waitKey(1)
        plt.plot(peak_x, self.histogram[peak_x], 'r+', markersize=8)
        plt.show()
    
    def visualize_deleted_click(self, x, y):
        plt.plot(x, y, 'r+', markersize=8)
        plt.show()

    def visualize_lane_points(self, img, lanes):
        cnt = 0
        for lane in lanes:
            for point in lane:
                cv2.circle(img, (point[0], point[1]), 3, self.colours[cnt], -1)
            cnt += 1 
        cv2.imshow("src", img)
        cv2.waitKey(0)

    def print_img_and_histogram(self, img, height):
        i = (height - self.bottom_row_index)// self.step % len(self.colours)
        color = (self.colours[i][2]/255, self.colours[i][1]/255, self.colours[i][0]/255)
        img = cv2.line(img, (0,height), (img.shape[1],height), self.colours[i]) 
        plt.figure(figsize=(15, 5))
        plt.plot(list(range(img.shape[1])), self.histogram , color=color)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Histogramm of the image slice')
        # Connect the click & key event 
        plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
        plt.gcf().canvas.mpl_connect('key_press_event', self.on_key)

        for point in self.histograms_peaks[height] : 
            self.visualize_point_on_image(point, height, False)

        cv2.imshow("src", img)
        cv2.waitKey(1)
        plt.show()
        # plt.clf()

    def compute_heights_of_histogram(self, img_height):
        self.bottom_row_index = img_height - self.bottom_offset
        end = int((1 - self.bottom_perc) * img_height)
        self.step = int(-(img_height * self.bottom_perc / self.slices))
        real_slices = int((end - self.bottom_row_index) // self.step)
        self.top_row_index = self.bottom_row_index + real_slices * self.step 

    def load_source(self):
        """Load from file and Save image to self.src and returns true if its successful else false 
            >>> prints error message if it fails"""

        if self.image_path:
            self.src = cv2.imread(self.image_path)
            self.image_path = None
            self.LD = LaneDetection(src.shape[1], src.shape[0], "455", None)

        elif self.video_path:
            # offset for the first frame
            # initialize video capture
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
                ret, src = self.cap.read()
                self.LD = LaneDetection(int(src.shape[1]), int(src.shape[0] ), "455", None)

                for _ in range(self.skip_first_frames - 1):
                    ret, src = self.cap.read()
                if not ret:
                    print(">>> Error: Could not read video")
                    self.cap.release()
                    self.cap = None
                    self.video_path = None
                    return False
            else:   
                for _ in range(self.skipped_frames):
                    ret, src = self.cap.read()
                    if not ret:
                        print(">>> End of video stream")
                        self.cap.release()
                        self.cap = None
                        self.video_path = None
                        return False 
            # src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
            self.src = src
        else : 
            return False 

        if self.src is None:
            print(">>> Error: Could not load image or video")
            return False
        return True


def main():
    print(">>> PHASE #1 : Detect peaks using LD ----------")
    print("----------- Lane Detection checking -----------")
    print(">>> Press 'y' to ACCEPT the detected peak,")
    print(">>> Press 'n' to DISMISS the detected peak,")
    print(">>> Press '0' to CONTINUE without auto-detection,")
    print(">>> Press 'q' to SKIP THIS FRAME.")
    print(">>> Press '!' to terminate the program.")

    print("\n>>> PHASE #2 : Detect points by hand ----------")
    print("----------- Lane Peak Labeling Tool -----------")
    print(">>> Press 'n' or ' ' to go to the NEXT slice,")
    print(">>> Press 'b' to go BACK to the PREVIOUS slice,")
    print(">>> Press 'r' to REMOVE the previous click,")
    print(">>> Press 'd' to DELETE the detected last peak,")
    print(">>> Press 'q' to SKIP THIS FRAME.") 
    print(">>> Press 'c' and click to add a peak at that point.") 
    print(">>> Press '!' to terminate the program.")

    print("\n>>> PHASE #3 : Classify points into lanes -----")
    print("Press '1-9' to select lane number")
    print("Press 'n' to go to the NEXT point")
    print("Press 'b' to go BACK to the PREVIOUS point")
    print()
    # Change these paths as needed
    # MUST HAVE
    video_path = "video_repository/real_roads/IMG_2997.mp4"
    # Interesting video
    video_path = "video_repository/real_roads/IMG_2988.mp4"
    # Interesting video
    video_path = "video_repository/real_roads/IMG_2987.mp4"
    # SO SO 
    video_path = "video_repository/real_roads/IMG_2986.mp4"
    # NO GOOD 
    video_path = "video_repository/real_roads/IMG_2985.mp4"
    # Highway Thessaloniki
    video_path = "video_repository/real_roads/IMG_2984.mp4"
    # THIS IS GOOD 
    video_path = "video_repository/real_roads/IMG_2983.mp4"
    # THIS IS GOOD 
    video_path = "video_repository/real_roads/IMG_2951.mp4"
    # THIS IS GOOD 
    video_path = "video_repository/real_roads/IMG_2950.mp4"
    # Highway from thessaloniki to Alexandroupoli
    video_path = "video_repository/real_roads/IMG_2948.mp4"
    video_path = "video_repository/real_roads/IMG_2946.mp4"
    video_path = "video_repository/real_roads/IMG_2944.mp4"
    # video_path = "video_repository/real_roads/IMG_2893.mp4"     # Almost DONE
    # video_path = "video_repository/real_roads/IMG_2892.mp4"     # Almost DONE
    # video_path = "video_repository/real_roads/IMG_2891.mp4"     # DONE
    # video_path = "video_repository/real_roads/IMG_2890.mp4"     # DONE
    # video_path = "video_repository/real_roads/IMG_2889.mp4"     # DONE
    # video_path = "video_repository/real_world.mp4"              # DONE 
    # video_path = "video_repository/real_roads/IMG_2888.mp4"     # DONE 
    labeler = LanePeakLabeler(video_path=video_path)

    # image_path = "image_repository/real_world.jpg"
    # labeler = LanePeakLabeler(image_path=image_path)
    labeler.start_labeling()
    
    print("Labeling complete!")


if __name__ == "__main__":
    main()