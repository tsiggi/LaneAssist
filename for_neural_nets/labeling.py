from data_helper import save_annotated_data
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting


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
        self.bottom_offset = 20 # in pixels
        self.bottom_perc = 0.58 # percentage of the height
        self.slices = 30        # number of slices
        self.colours = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
            (128, 0, 128), (0, 128, 128), (128, 128, 0)
        ]
        self.skipped_frames = 100 # Number of frames to skip in video
        self.height_minimum_difference = 2 # Minimum height difference to consider a peak
        
        # Initialize data structures
        self.histograms_peaks = {}  # Key: slice height, Value: list of widths 
        self.lanes = None # store lists of lane points depending on the self.max_lanes 

        # Helpful variables
        self.cap = None # VideoCapture object
        self.clicks = []
        self.flag_go_back = False
        self.flag_skip_this_frame = False

    def start_labeling(self):
        # load the source [image or frame of a video (repeat for every frame)]
        while self.load_source(): 
            image_for_saving = self.src.copy() 
            frame = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
            self.compute_heights_of_histogram(frame.shape[0])

            # In a while loop cause we need to change self.height when the user goes back (to the previous slice)
            # bottom > top --->  step < 0
            self.height = self.bottom_row_index
            while self.height > self.top_row_index - 1:
                self.clicks = [] # reset the clicks for each slice
                self.histograms_peaks[self.height] = [] # init for this slice
                
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

    def reset_params(self):
        self.histograms_peaks = {}
        self.lanes = None
        self.flag_go_back = False

    def classify_into_lanes(self):
        """Phase 2: Classify detected points into lanes"""
        
        
        def wait_for_key():
            while True:
                key = cv2.waitKey(0)

                if key == -1:
                    continue

                key_char = chr(key & 0xFF)  # convert to character
                
                if key_char in ['1','2','3','4','5','6','7','8','9']:
                    current_lane = int(key_char) - 1 
                    cv2.circle(self.classification_image, (x, y), 3, self.colours[current_lane], -2)
                    cv2.circle(self.src, (x, y), 3, self.colours[current_lane], -2)
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
                height = height - self.step if height < self.bottom_row_index else self.bottom_row_index
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

    def find_peak_from_user_clicks(self, left_x, left_y, right_x):
        i = left_x
        step = 3
        cnt = 0 
        while (self.histogram[i] - left_y < self.height_minimum_difference) and (i < right_x):
            i+= step
            cnt += 1
        if cnt < 2 or i > right_x:
            return None
        x_1 = i - step / 2
        cnt = 0
        while (self.histogram[i] - left_y > self.height_minimum_difference) and (i < right_x):
            i+= step
            cnt += 1
        if cnt < 2 or i > right_x:
            return None
        x_2 = i - step / 2
        return int((x_1 + x_2) / 2)

        while (self.histogram[i] - self.height_minimum_difference < left_y and self.histogram[i+2] - self.height_minimum_difference > left_y) and (i+2 < right_x):
            i+= 2 
        if i + 2 > right_x:
            return None
        x_1 = i 
        i+= 2 
        while (self.histogram[i] + self.height_minimum_difference > left_y and self.histogram[i+2] + self.height_minimum_difference < left_y) and (i+2 < right_x):
            i+= 2
        if i + 2 > right_x:
            return None     
        x_2 = i
        print (x_1, x_2)
        return int((x_1 + x_2) / 2)

    def on_click(self, event):
        """Handle mouse clicks on the histogram plot"""
        if event.xdata is None or event.ydata is None:
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
            
            # Ensure left_x is actually to the left
            if left_x > right_x:
                left_x, right_x = right_x, left_x

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
        
        elif event.key == '!': # Terminate 
            print(">>> Exiting...")
            exit(1)

    def visualize_hist_peaks(self):
        for peaks in self.histograms_peaks[self.height]:
            peak_x = peaks
            cv2.circle(self.src, (peak_x, self.height), 3, (255, 0, 255), 3)
            cv2.imshow("src", self.src)
            cv2.waitKey(1)

    def visualize_point_on_image(self, peak_x, current_height):
        cv2.circle(self.visualized_frame, (peak_x, current_height), 3, (230, 216, 173), 3)
        cv2.imshow("src", self.visualized_frame)
        cv2.waitKey(1)
        plt.plot(peak_x, self.histogram[peak_x], 'go', markersize=8)
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

        cv2.imshow("src", img)
        cv2.waitKey(1)
        
        plt.plot(list(range(img.shape[1])), self.histogram , color=color)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Histogramm of the image slice')
        # Connect the click & key event 
        plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
        plt.gcf().canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()
        plt.clf()   

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
            
        elif self.video_path:
            # initialize video capture
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
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
            self.src = src
        else : 
            return False 

        if self.src is None:
            print(">>> Error: Could not load image or video")
            return False
        return True


def main():
    print("----------- Lane Peak Labeling Tool -----------")
    print(">>> Press 'n' or ' ' to go to the NEXT slice,")
    print(">>> Press 'b' to go BACK to the PREVIOUS slice,")
    print(">>> Press 'r' to REMOVE the previous click,")
    print(">>> Press 'd' to DELETE the detected last peak,")
    print(">>> Press 'q' to SKIP THIS FRAME.") 
    print(">>> Press '!' to terminate the program.")
    print("\n------ For classifing points into lanes: ------")
    print("Press '1-9' to select lane number")
    print("Press 'n' to go to the NEXT point")
    print("Press 'b' to go BACK to the PREVIOUS point")
    print()
    # Change these paths as needed
    video_path = "video_repository/real_world.mp4"
    labeler = LanePeakLabeler(video_path=video_path)

    # image_path = "image_repository/real_world.jpg"
    # labeler = LanePeakLabeler(image_path=image_path)
    labeler.start_labeling()
    
    print("Labeling complete!")


if __name__ == "__main__":
    main()