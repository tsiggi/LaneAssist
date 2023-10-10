from src.LaneKeeping.lanekeeping import LaneKeeping
from src.LaneDetection.detect import LaneDetection
from src.variablesHandler import VariablesHandler
import configparser
import logging
import imageio
import time
import cv2
import numpy as np


if __name__ == "__main__" :

    config = configparser.ConfigParser()
    config.read("config.ini")

    video = config["GENERAL"].getboolean("video")

    # Test Video
    if video :
        
        cap = cv2.VideoCapture("video_repository/highway.mp4")
        # cap = cv2.VideoCapture("video_repository/intersection-road-block.mp4")
        # cap = cv2.VideoCapture("video_repository/intersections.mp4")
        # cap = cv2.VideoCapture("video_repository/intersections2.mp4")
        
        # cap = cv2.VideoCapture("video_repository/real_world.mp4")
        real_world_example = False # make it true if you use the real_world.mp4 video !!!
        # better results with increased number of slices 

        skipped_frames = 10 #@param {type:"integer"}
        frames = []
        print("Keys :\n- ' ' = Pause/Unpause Video (Spacebar)\n- 'q' = Quit Video\n- 's' = Start saving frames inorder to create a gif\n- 'e' : End/Stop saving frames for the output gif")
        
        # Parsing of the video
        font = cv2.FONT_HERSHEY_SIMPLEX
        sf = -1
        log = logging.getLogger('Root logger')
        start_saving_frames = False
        
        cnt = -1
        frames_used = 0
                        
        time_sum = 0
        while(cap.isOpened()):  
                        
            ret, src = cap.read()

            if ret :

                # initialize them
                if cnt<0 :
                    camera = "405"
                    if real_world_example :
                        camera = "455"
                    ld = LaneDetection(src.shape[1], src.shape[0], camera)
                    lk = LaneKeeping(src.shape[1], src.shape[0], log, camera)
                    if real_world_example :
                        ld.square_pulses_min_height = 80
                        ld.square_pulses_pix_dif = 10
                        ld.square_pulses_min_height_dif = 20
                        ld.square_pulses_allowed_peaks_width_error = 15

                if real_world_example or cnt % skipped_frames == 0 :
                    frames_used += 1
                    start = time.time()
                    results = ld.lanes_detection(src)
                    end = time.time()
                    elapsed = end - start
                    time_sum += elapsed 
                    angle, src = lk.lane_keeping(results)
                    cv2.imshow('lk',src)
                    cv2.waitKey(1)

                    if start_saving_frames :
                        cv2.imwrite(f".frames/lk_frame{frames_used}.jpg", src)
                        # frames.append(src)

                cnt+=1
                
                key = cv2.waitKey(100)
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

        src = cv2.imread("image_repository/test_real_world.jpg")
        log = logging.getLogger('Root logger')

        # src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
        ld = LaneDetection(src.shape[1], src.shape[0], "405")
        lk = LaneKeeping(src.shape[1], src.shape[0], log, "405")
        

        results = ld.lanes_detection(src.copy())
        cv2.imshow('ld', results["frame"])
        angle1, src1 = lk.change_lane_maneuver(results, direction="left", overtake_type="static")
        results = ld.lanes_detection(src.copy())
        angle2, src2 = lk.change_lane_maneuver(results, direction="right", overtake_type="dynamic")
        results = ld.lanes_detection(src.copy())
        lk.reset_change_lane_params()
        angle3, src3 = lk.lane_keeping(results)

        # cv2.imwrite("frame2.jpg",src)

        cv2.imshow('Turn_Left.jpg', src1)
        cv2.imshow('Turn_Right.jpg', src2)
        cv2.imshow('lk_frame.jpg', src3)
        cv2.waitKey(0)
        