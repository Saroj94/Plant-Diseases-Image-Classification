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
## detect corrupted images and delete permanently

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

## Instead of deleting immediately, we can move it
# Move corrupted images to a separate folder 
# Shallow corrupted image checker function
<!-- 
def move_corrupted_images(image_dir, bad_dir):

    ##creates the folder where all the corrupted files will be dump
    os.makedirs(bad_dir, exist_ok=True)

    ##os.walk(): traverses a directory tree 
    ##where you are (root)
    #which subfolders exist there (dirs)
    #which files exist there (files)
    for root, dirs, files in os.walk(image_dir):
        ##iterate through files
        for file in files:
            ##search if files name ends with these extension
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                ##if yes then join the existing path
                path = os.path.join(root, file)
                try:
                    ##open the path of the image
                    img = Image.open(path)
                    ##checks inconsistency
                    img.verify()
                except:
                    shutil.move(path, os.path.join(bad_dir, file)) -->


# Hardcode data spliting 
# ##Identify class labels
<!-- # folder_name_list=os.listdir(RAW_DIR)

# ##Folder names become class labels
# ##keeps only folders ["apple", "mango", "grapes", "potato"]
# classes = [cls for cls in folder_name_list if os.path.isdir(os.path.join(RAW_DIR, cls))] 

# # Create directory structure
# ##Create output directory structure
# ##Iterates over dataset splits
# for split in ["train", "val", "test"]:
#     ##Iterates over class labels
#     for clss in classes:
#         os.makedirs(os.path.join(OUT_DIR, split, clss), exist_ok=True) #Creates directories

# # Split per class
# #Process one class at a time
# #This ensures class-wise stratification
# for cls in classes:
#     #full path to one class folder like data/raw/apple
#     cls_path = os.path.join(RAW_DIR, cls)
#     ##list of images within list of folder 
#     img_list=os.listdir(cls_path)
#     #Collects only image files
#     images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]

#     ##image suffling
#     random.shuffle(images)

#     #Compute split sizes
#     n_total = len(images)
#     n_train = int(TRAIN_RATIO * n_total)
#     n_val = int(VAL_RATIO * n_total)

#     train_imgs = images[:n_train]
#     val_imgs = images[n_train:n_train + n_val]
#     test_imgs = images[n_train + n_val:]

#     #Copy images into split folders
#     for img in train_imgs:
#         shutil.copy(
#             os.path.join(cls_path, img),
#             os.path.join(OUT_DIR, "train", cls, img)
#         )

#     for img in val_imgs:
#         shutil.copy(
#             os.path.join(cls_path, img),
#             os.path.join(OUT_DIR, "val", cls, img)
#         )

#     for img in test_imgs:
#         shutil.copy(
#             os.path.join(cls_path, img),
#             os.path.join(OUT_DIR, "test", cls, img)
#         )

# print("Train / Validation / Test split completed.") -->


# Tensorflow function to split data
<!-- # ##data directories
# TRAIN_DIR='data/train'
# TEST_DIR='data/test'
# VAL_DIR='data/val'

# # Loads images from directories, automatically assigns labels from folder names
# ##output dataset format look like this (images,labels) in memory
# train_ds = tf.keras.utils.image_dataset_from_directory( 
#     TRAIN_DIR,
#     image_size=(IMG_HIGHT,IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     shuffle=True,       # Shuffle training dataset
#     seed=42,
#     validation_split=False
# )

# ##validation data preprocess
# val_ds = tf.keras.utils.image_dataset_from_directory( 
#     VAL_DIR,
#     image_size=(IMG_HIGHT,IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     shuffle=False,       #shuffle is not needed for validation/test
#     validation_split=False
# )

# ##test data preprocess
# test_ds = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_HIGHT,IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     shuffle=False,    #shuffle is not needed for validation/test
#     validation_split=False
# ##disease class
# dis_class=train_ds.class_names
# for x, y in train_ds.take(1):
#     print(type(x), x.shape)
# ) -->


# Corrupted file removable function but not worked
<!-- # ##cleaning data
# def clean_corrupt_dataset_checker(image_dir, bad_dir):
#     # Create the folder where corrupted images will be stored.
#     os.makedirs(bad_dir, exist_ok=True)

#     # Walk through all directories and files inside image_dir.(Walking through the dataset)
#     # os.walk returns:
#     #   root  -> current folder path
#     #   dirs  -> list of subfolders inside root
#     #   files -> list of files inside root
#     for root, dirs, files in os.walk(image_dir):

#         # Loop through every file found in the current folder
#         for file in files:

#             # Process only image files with these extensions
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):

#                 # Build the full path to the current image file
#                 src_path = os.path.join(root, file)

#                 # Start by assuming the image is clean
#                 corrupted = False

#                 # ----------------------------------------------------
#                 # Pillow validation (two-step check)
#                 # ----------------------------------------------------
#                 try:
#                     # First check: verify the image header structure
#                     with Image.open(src_path) as img:
#                         '''
#                         The file type (JPEG, PNG, etc.)
#                         The image width and height
#                         The color mode (RGB, grayscale, etc.)
#                         Compression details
#                         Metadata markers
#                         '''
#                         img.verify()

#                     # Second check: fully load the image pixel data
#                     with Image.open(src_path) as img:
#                         img.load()

#                 except Exception:
#                     # If Pillow fails at any point, mark as corrupted
#                     corrupted = True

#                 # TensorFlow validation (strict decoder)
#                 # ----------------------------------------------------
#                 if not corrupted:  # Only check if Pillow passed
#                     try:
#                         '''
#                         Pillow checks:
#                                 To catch header issues and partial corruption.
#                         TensorFlow checks:
#                                 To catch strict JPEG/PNG decoding errors that Pillow misses.
#                         '''
#                         # Read the image file as raw bytes
#                         #TensorFlow can only decode images from raw bytes, not from a Pillow image object or a file path.
#                         img_bytes = tf.io.read_file(src_path)

#                         # Try decoding it using TensorFlow's strict decoder which make images compatible with tensorflow
#                         #If decoding succeeds → the image is clean.
#                         #If decoding fails → the image is corrupted.
#                         img1=tf.image.decode_jpeg(img_bytes, channels=3)

#                         # FORCE resize (this catches hidden corruption)
#                         img2 = tf.image.resize(img1, (IMG_HEIGHT, IMG_WIDTH))

#                         # Force full materialization
#                         img = tf.cast(img2, tf.uint8)

#                     except Exception:
#                         # If TensorFlow fails, the image is corrupted
#                         corrupted = True

#                     # Move the file to the appropriate folder
#                     # ----------------------------------------------------
#                     if corrupted:
#                         # Move corrupted image to bad_dir
#                         shutil.move(src_path, os.path.join(bad_dir, file)) -->