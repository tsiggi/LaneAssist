from data_helper import load_annotated_data
from dataset_augmentation import DatasetAugmentation

DATASET_DIR = "final_dataset_2"
CREATE_DATASET = False           # False if dataset already exists 

# FROM 271 FULLY ANNOTATED IMAGES :  
# "final_dataset"    :  5.067 different labels 
# "final_dataset_2"  : 18.408 different labels [USE THIS ONE]
# "final_dataset_3"  : 26.164 different labels 

if CREATE_DATASET:
    dataset = load_annotated_data(base_dir="lane_anotation_dataset")
    data_prep = DatasetAugmentation(dataset)
    data_prep.split_dataset_and_save_it(DATASET_DIR)

data_prep = DatasetAugmentation()
images, labels = data_prep.get_dataset(DATASET_DIR)

# dataset = data_prep.load_annotated_data(DATASET_DIR) # [NOT MEMORY EFFICIENT] Stores the same images multiple times

for i, labels in enumerate(labels):
    data_prep.visualize_image_with_label(images[labels['image_id']], labels['lane_points'], labels['step'], labels['hists_num'])

