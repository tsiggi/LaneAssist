import cv2
import os
import pickle
from datetime import datetime


def save_annotated_data(image, lane_points, lane_assignments, output_dir="lane_annotation_dataset"):
    """
    Save an annotated image and its lane labels
    
    Parameters:
    - image: The original image (numpy array)
    - lane_points: Dictionary with keys the height and values the width of the points
    - lane_assignments: list of lists of points tuples (height, width)
    """

    # MAYBE CHANGE lane_assignments to that : 
    # Dictionary of lane lists, key is the number of the lane and value is a list of points tuples(height, width)

    # Create directories if they don't exist
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    for directory in [output_dir, images_dir, labels_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the image
    image_filename = os.path.join(images_dir, f"lane_image_{timestamp}.jpg")
    cv2.imwrite(image_filename, image)
    
    # Create data structure to save
    data = {
        'image_filename': os.path.basename(image_filename),
        'lanes': lane_assignments,
        'raw_lane_points': lane_points
    }
    
    # Save the lane data using pickle
    label_filename = os.path.join(labels_dir, f"lane_data_{timestamp}.pkl")
    with open(label_filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Image saved to: {image_filename}")
    print(f"Labels saved to: {label_filename}")
    
    return image_filename, label_filename

def load_annotated_data(base_dir="lane_anotation_dataset"):
    """Load all saved annotation data from custom directory structure"""
    all_data = []
    
    data_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.pkl')]
    
    for label_file in label_files:
        # Load the label data
        with open(os.path.join(labels_dir, label_file), 'rb') as f:
            data = pickle.load(f)
        
        # Load the corresponding image
        image_path = os.path.join(data_dir, data['image_filename'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # Add to our dataset
            all_data.append({
                'image': image,
                'lanes': data['lanes'],
                'raw_points': data['raw_lane_points']
            })
        else:
            print(f"Warning: Image {image_path} not found")
    
    print(f"Loaded {len(all_data)} annotated samples")
    return all_data