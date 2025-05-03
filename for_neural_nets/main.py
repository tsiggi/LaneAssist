from data_helper import load_annotated_data
import cv2

colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (128, 0, 255)]

dataset = load_annotated_data(base_dir="lane_anotation_dataset")

for data in dataset:

    image = data['image']
    lanes = data['lanes']
    lane_points = data['raw_points']

    # Visualize the lanes
    for i, lane in enumerate(lanes): 
        for point in lane:
            cv2.circle(image, (point[0], point[1]), 5, colours[i], 5)
    
    cv2.imshow("Lanes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



