"""
MIT License 
Copyright (c) 2023 Christos-Alexandros Tsingiropoulos

This software contains work that was developed by Christos-Alexandros Tsingiropoulos
as part of the Bosch Future Mobility Challenge (BFMC) competition project, which
was organized by Bosch. This work represents Christos-Alexandros Tsingiropoulos's
individual contribution to the larger project created by the VROOM team.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3) THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

4) While not mandatory, it would be appreciated if you include the following attribution
when using or showcasing the work contributed by Christos-Alexandros Tsingiropoulos:

"VROOM - BFMC 2023 Competition Project"
Contributed by Christos-Alexandros Tsingiropoulos

This attribution is appreciated as it gives credit to the team and individual contributors
who worked on this project.
"""

from src.LaneKeeping.lanekeeping import LaneKeeping
from src.LaneDetection.detect import LaneDetection
import configparser
import logging
import time
import cv2


if __name__ == "__main__" :

    config = configparser.ConfigParser()
    config.read("config.ini")

    video = config["GENERAL"].getboolean("video")

    # Test Video
    if video :
        
        cap = cv2.VideoCapture("video_repository/highway.mp4")
        cap = cv2.VideoCapture("video_repository/intersection-road-block.mp4")
        
        # cap = cv2.VideoCapture("video_repository/real_world.mp4")
        real_world_example = False # make it true if you use the real_world.mp4 video !!!

        skipped_frames = 10 #@param {type:"integer"}
        frames = []
        print("Keys :\n- ' ' = Pause/Unpause Video (Spacebar)\n- 'q' = Quit Video\n- 's' = Start saving frames inorder to create a gif\n- 'e' : End/Stop saving frames for the output gif")
        
        # Parsing of the video
        font = cv2.FONT_HERSHEY_SIMPLEX
        sf = -1
        log = logging.getLogger('Root logger')
        start_saving_frames = False
        # video = cv2.VideoWriter("video.avi", 0, 60, (width,height))
        
        cnt = -1
        frames_used = 0
                        
        time_sum = 0
        while(cap.isOpened()):  
                        
            ret, src = cap.read()
            if ret :
                # src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
    
                # initialize them
                if cnt<0 :
                    camera = "455"
                    lk = LaneKeeping(src.shape[1], src.shape[0], log, camera)
                    ld = LaneDetection(src.shape[1], src.shape[0], camera, lk)
                    if real_world_example :
                        # params adjusted for real roads
                        ld.square_pulses_min_height = 80
                        ld.square_pulses_pix_dif = 10
                        ld.square_pulses_min_height_dif = 20
                        ld.square_pulses_allowed_peaks_width_error = 15
                        # also change (peaks_min/max_width, bottom_perc, bottom_width, top_width)

                if real_world_example or cnt % skipped_frames == 0 :
                    frames_used += 1
                    start = time.time()
                    ld_frame = src.copy()
                    results = ld.lanes_detection(ld_frame)
                    # hor_frame = src.copy()
                    # hor_line, line, hor_exists = ld.horizontal_detection(hor_frame)
                    # cv2.imshow('hor2', hor_frame)
                    # if hor_exists :
                    #     ld.visualize_horizontal_line(results['frame'], hor_line, line)
                    end = time.time()
                    elapsed = end - start
                    time_sum += elapsed 
                    angle, src = lk.lane_keeping(results)
                    cv2.imshow('lk',src)
                    cv2.waitKey(1)

                    if start_saving_frames :
                        cv2.imwrite(f".frames/ld_frame/{frames_used}.jpg", ld_frame)
                        # cv2.imwrite(f".frames/hor_frame/{frames_used}.jpg", hor_frame)
                cnt+=1
                
                key = cv2.waitKey(10)
                # Exit Video
                if key == ord('q') :
                    print("Exiting...")
                    break
                # Pause Video (and make exit work while paused)
                if key == ord(' '):
                    print("Pause")
                    key2 = 0
                    while True:
                        key2 = cv2.waitKey(100)
                        if key2 == ord(' '):
                            print("Unpause")
                            break
                        elif key2 == ord('q'):
                            print("Exiting...")
                            break
                    if key2 == ord('q'):
                        break
                if key == ord('s'):
                    start_saving_frames = True
                    print("Start Saving frames ...")
                if key == ord('e'):
                    if start_saving_frames :
                        start_saving_frames = False
                        print("End of saving frames...")
                
            else : 
                break
        
        if cnt>0:
            print(f"\nAverage time: {time_sum/frames_used}s\n\n")  
    
    # Test Photos
    else :

        test_lane_detection = True
        test_horizontal = False
        test_change_lane = False

        log = logging.getLogger('Root logger')
        
        if test_lane_detection :
            src = cv2.imread("image_repository/real_world.jpg")
            # src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)

            # Initialize ld and lk
            ld = LaneDetection(src.shape[1], src.shape[0], "455")
            lk = LaneKeeping(src.shape[1], src.shape[0], log, "455")
            # get results for that frame
            results = ld.lanes_detection(src.copy())
            angle, f = lk.lane_keeping(results)
            
            cv2.imshow("ld", f)
            cv2.imwrite("ld.jpg", results["frame"])
            cv2.waitKey(0)
        if test_horizontal :
            # Get image and rotate it
            src = cv2.imread("image_repository/Starting_point.png")
            src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
            rotation_angle = 10
            height, width = src.shape[:2]
            center = (width/2, height/2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle, scale=1)
            rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
            src = rotated_image

            # Initialize ld and lk
            ld = LaneDetection(src.shape[1], src.shape[0], "455")
            lk = LaneKeeping(src.shape[1], src.shape[0], log, "455")

            # get results for that frame
            results = ld.lanes_detection(src.copy())
            cv2.imshow('ld', results["frame"])
            hor_frame = src.copy()
            hor_line, line, hor_exists = ld.horizontal_detection(hor_frame)
            if hor_exists :
                ld.visualize_horizontal_line(results['frame'], hor_line, line)
            cv2.imshow('hor', results["frame"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if test_change_lane :
            src = cv2.imread("image_repository/test_real_world.jpg")
            
            ld = LaneDetection(src.shape[1], src.shape[0], "405")
            lk = LaneKeeping(src.shape[1], src.shape[0], log, "405")
                    
            # LD params adjusted for this real road example
            ld.square_pulses_min_height = 150
            ld.square_pulses_pix_dif = 10
            ld.square_pulses_min_height_dif = 60
            ld.square_pulses_allowed_peaks_width_error = 15

            # show to visualized results
            results = ld.lanes_detection(src.copy())
            # cv2.imshow('ld', results["frame"])
            angle1, src1 = lk.change_lane_maneuver(results, direction="left", overtake_type="static")
            results = ld.lanes_detection(src.copy())
            angle2, src2 = lk.change_lane_maneuver(results, direction="right", overtake_type="dynamic")
            results = ld.lanes_detection(src.copy())
            lk.reset_change_lane_params()
            angle3, src3 = lk.lane_keeping(results)

            cv2.imshow('Turn_Left.jpg', src1)
            cv2.imshow('Turn_Right.jpg', src2)
            cv2.imshow('lk_frame.jpg', src3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        