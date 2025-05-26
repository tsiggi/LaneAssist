from data_helper import save_image, save_labels
import pickle
import cv2
import os

WIDTH = 960
HEIGHT = 540
OUTPUT_DIR = "final_dataset_2"
WAIT = 100 # milliseconds 
COLOURS = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 128, 0),     # Orange
    (128, 0, 255),     # Purple
    (0, 128, 255),     # Sky Blue
    (128, 255, 0)      # Lime
]

class DatasetAugmentation: 
    def __init__(self, dataset=None, img_width=WIDTH, img_height=HEIGHT):
        self.dataset = dataset
        self.dataset_len = len(dataset) if dataset is not None else 0  # Number of images and annotations in the dataset
        self.img_width = img_width      # for the resizing
        self.img_height = img_height    
        self.number_of_data_after_split = 0
        self.subset_id = 0
    
    def split_dataset_and_save_it(self, output_dir=OUTPUT_DIR):
        if self.dataset is None:
            raise ValueError("Dataset is not provided. Please provide a dataset to augment.")

        for i in range(self.dataset_len):
            data = self.dataset[i]          # get data

            img, lanes_points = self.resize_image(data['image'], data['points_with_lanes'], self.img_width, self.img_height)

            splitted_data = self.split_annotations(lanes_points)

            self.save_image_and_anotation(img, splitted_data, data['image_filename'], output_dir)
        
        print(f">>> Dataset augmentation completed. From {self.dataset_len} images, we created {self.number_of_data_after_split} different labels.")

    def save_image_and_anotation(self, image, splitted_data, image_name=None, output_dir=OUTPUT_DIR): 
        
        # First save the image (size: WIDTH x HEIGHT)
        save_image(image, output_dir=output_dir, filename=image_name)

        self.subset_id = 0
        # For each subset of lane points, save the labels in a different file
        for lane_points in splitted_data:
            self.subset_id += 1 
            keys = list(lane_points.keys())
            step = keys[1] - keys[0] if len(keys) > 0 else None
            filename= f"img_{image_name[11:-4]}_label_subset_{self.subset_id}.pkl" 
            save_labels(image_name, lane_points, step, len(lane_points), output_dir, filename)

    def split_annotations(self, lane_points):
        """ Create subsets of histograms, each subset has a constant step on the height axis,
            different step or different starting height.
            
            return a list of different sets of lane points
        """

        splitted_data = [] 
        x = 12 
        hist_num = len(lane_points.keys())
        max_step = hist_num // x
        
        list_of_keys = list(lane_points.keys())
        
        # 0,1,2,..., hist_num - x
        for step in range(1, max_step + 1):

            for starting_hist in range(0, hist_num//3 - step):

                set = {}

                for hist_id in range(starting_hist, hist_num, step):

                    if hist_id < len(list_of_keys):
                        key = list_of_keys[hist_id]
                        set[key] = lane_points[key]
                
                if len(set) > x: 
                    splitted_data.append(set)
                    self.number_of_data_after_split += 1

        return splitted_data


        # i stores the step => and thus the number of different sets that can be extracted from the histograms
        # i = 1 step = 1 => 1 set (with size hist_num)
        # i = 2 step = 2 => 2 sets (with size hist_num/2)
        # etc.
        for i in range(1, max_step + 1): 

            for j in range(i):
                set = {} # store the data for the current step

                for k in range(hist_num//i):
                    start = j 
                    step = i

                    cnt = 0 
                    for key in lane_points: 
                        if (cnt - start) % step == 0:
                            set[key] = lane_points[key]
                            cnt += 1
                        else:
                            cnt += 1
                            continue
                splitted_data.append(set)
                self.number_of_data_after_split += 1

        return splitted_data
        
    def resize_image(self, image, lane_points, width, height):
        """Resize image to the specified width and height, 
           and adjust lane points accordingly.
        """

        # if image has the correct size, return  
        if image.shape[0] == height and image.shape[1] == width:
            return image, lane_points
        
        # Resize the image to the specified width and height
        img_h, img_w = image.shape[:2]
        image = cv2.resize(image, (width, height))

        # Addapt the lane points to the new image size
        l_points = {}
        for i in lane_points.keys():
            # i is the point height
            k = int(i * height / img_h)
            l_points[k] = {} 
            for j in lane_points[i].keys():
                # j is the point width
                l_points[k][int(j * width/ img_w)] = lane_points[i][j] 
        
        return image, l_points 

    def load_annotated_data(self, base_dir=OUTPUT_DIR):
        """Load all saved annotation data from custom directory structure.
        Same images are stored multiple times, because of the different labels.

        Returns:
            all_data: a list of dictionaries, each having as keys {image, lane_points, step, hists_num}
        """
        print(f">>> Loading annotated data from {base_dir} ...")
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
                    'lane_points': data['lane_points'],
                    'step': data['step'] if 'step' in data else None,
                    'hists_num': data['hists_num'] if 'hists_num' in data else None
                })
            else:
                print(f"Warning: Image {image_path} not found")
        
        print(f">>> DONE! Loaded {len(all_data)} annotated samples.")
        return all_data

    def get_dataset(self, base_dir=OUTPUT_DIR):
        """Load annotated data from the specified directory. 
        DOES NOT STORE THE SAME IMAGE TWISE, different labels for the same image are stored in different files.
        SO THE SIZES OF THE IMAGES AND LABELS ARE NOT EQUAL.

        Returns: 
            images: a list of all the images,
            labels: a list of dictionaries each having as keys {image_id, lane_points, step, hists_num}
        """
        print(f">>> Loading annotated data from {base_dir} ...")
        images, labels = [], []
        img_name_to_list_id = {}
        
        data_dir = os.path.join(base_dir, "images")
        labels_dir = os.path.join(base_dir, "labels")
        
        # Get all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.pkl')]
        
        for label_file in label_files:
            # Load the label data
            with open(os.path.join(labels_dir, label_file), 'rb') as f:
                data = pickle.load(f)
            # Check if image is already loaded to the 
            if not data['image_filename'] in img_name_to_list_id:
                image_path = os.path.join(data_dir, data['image_filename'])
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    images.append(image)
                    img_name_to_list_id[data['image_filename']] = len(images) - 1 
                else:
                    print(f"Warning: Image {image_path} not found")
                    continue
            list_id = img_name_to_list_id[data['image_filename']]

            labels.append({
                'image_id': list_id,
                'lane_points': data['lane_points'],
                'step': data['step'] if 'step' in data else None,
                'hists_num': data['hists_num'] if 'hists_num' in data else None,
            })

        print(f">>> DONE! Loaded {len(labels)} annotated samples, from {len(images)} images.")
        return images, labels 

    def visualize_image_with_label(self, image, lane_points, step, hist_num, id=None):
        img = image.copy()
        cv2.putText(img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Histograms: {hist_num}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for j in lane_points.keys():
            img = cv2.line(img, (0,j), (img.shape[1],j), (128,128,128)) 
            for point in lane_points[j]:
                lane = lane_points[j][point]
                cv2.circle(img, (point, j), 2, COLOURS[lane], 2)
        cv2.imshow(f"Dataset", img)
        key = cv2.waitKey(WAIT)
        key_char = chr(key & 0xFF)  # convert to character
        if  key_char == 'q':
            cv2.destroyAllWindows()
            exit(0)
        elif key_char == ' ':
            cv2.waitKey(0)
 