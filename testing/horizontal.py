from src.LaneDetection.detect import LaneDetection
import configparser
import logging
import cv2


if __name__ == "__main__" :
    slope_0_25 = True
    slope_0_5 = True
    vizualize = True


    config = configparser.ConfigParser()
    config.read("config.ini")

    # Test Photos
    src = cv2.imread("image_repository/Starting_point.png")
    log = logging.getLogger('Root logger')

    src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
    
    rotation_angle = 15  # rotate image
    height, width = src.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle, scale=1)
    rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
    src = rotated_image
    cv2.imshow('Rotated Image', src)
    cv2.waitKey(3000)

    height, width = src.shape[0], src.shape[1]

    ld = LaneDetection(src.shape[1], src.shape[0], "405", None)
    ld.hor_step_by_step = True
    
    if slope_0_25 :
        cv2.destroyAllWindows()
        
        src1 = src.copy()
        hor_line, line ,hor_exists = ld.horizontal_detection(src1)
        cv2.imshow('Under the hook. Default allowed slope (0.25)', src1)
        # print(f"{hor_line} {line} {hor_exists}")

        if vizualize :   
            src3 = src.copy()
            ld.visualize_horizontal_line(src3, hor_line, line)
            cv2.imshow('Result visualization (0.25)', src3)

        cv2.waitKey(0)


    if slope_0_5 :
        cv2.destroyAllWindows()
        
        src2 = src.copy()
        hor_line, line ,hor_exists = ld.horizontal_detection(src2, max_allowed_slope=0.5)
        cv2.imshow('Under the hook. Allowed slope (0.5)', src2)
        # print(f"{hor_line} {line} {hor_exists}")
        
        if vizualize : 
            src3 = src.copy()
            ld.visualize_horizontal_line(src3, hor_line, line)
            cv2.imshow('Result visualization (0.5)', src3)
    
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()



    