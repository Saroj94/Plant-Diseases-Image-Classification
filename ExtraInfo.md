<!-- # import tensorflow as tf

Inbuilt Tensorflow function that creates train test data
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "raw/train",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     seed=42
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "raw/val",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     "raw/test",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# ) -->


## To Find corrupted images
<!-- from PIL import Image
import os

def find_corrupted_images(image_dir):
    corrupted = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()   # verifies file integrity
                except Exception as e:
                    corrupted.append((path, str(e)))
    
    return corrupted -->

## Search for corrupted images and delete them
##detect corrupted images and delete permanently

<!-- def delete_corrupted_images(image_dir):
    """	Scans an image dataset recursively
		Checks whether each image file is corrupted
		Deletes corrupted images automatically
    """
    #Recursively traverses all folders and subfolders and Returns three values for each directory
    for root, _, files in os.walk(image_dir):

        # Iterates over every file in the current directory
        for file in files:
            #Filtering only image files name endswith '.jpg', '.jpeg', '.png'
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                #Constructing the full file path
                ''' root = "data/images/class_1"
                    file = "img1.jpg"
                    path = "data/images/class_1/img1.jpg 
                '''
                path = os.path.join(root, file)
                try:
                    #Opening the image safely
                    img = Image.open(path)
                    ##Verifying image integrity
                    '''	•	File structure consistency
                        •	JPEG markers
                        •	End-of-file correctness
                        •	Compression format validity
                    '''
                    img.verify()
                except:
                    print("Deleting:", path)
                    os.remove(path) -->