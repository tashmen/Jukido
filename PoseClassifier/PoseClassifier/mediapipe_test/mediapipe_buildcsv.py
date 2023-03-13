from mediapipe_common import *

split = True
bootstrap = split
# Required structure of the images_in_folder:
#
#   fitness_poses_images_in/
#     pushups_up/
#       image_001.jpg
#       image_002.jpg
#       ...
#     pushups_down/
#       image_001.jpg
#       image_002.jpg
#       ...
#     ...
bootstrap_images_in_folder = 'E:/Jukido/Fun/Tensorflow/PythonScripts/jukido_stances'

# Output folders for bootstrapped images and CSVs.
bootstrap_images_out_folder = 'E:/Jukido/Fun/Tensorflow/PythonScripts/jukido_stances_out'
bootstrap_csvs_out_folder = bootstrap_images_out_folder

if split:
    split_into_train_test(bootstrap_images_in_folder, bootstrap_images_out_folder, 0.25)

# Initialize helper.
train_folder = '/train'
bootstrap_helper_train = BootstrapHelper(
    images_in_folder=bootstrap_images_out_folder + train_folder,
    images_out_folder=bootstrap_images_out_folder + train_folder,
    csvs_out_folder=bootstrap_csvs_out_folder + train_folder
)

test_folder = '/test'
bootstrap_helper_test = BootstrapHelper(
    images_in_folder=bootstrap_images_out_folder + test_folder,
    images_out_folder=bootstrap_images_out_folder + test_folder,
    csvs_out_folder=bootstrap_csvs_out_folder + test_folder
)

def bootstrap(bootstrap_helper):
    # Check how many pose classes and images for them are available.
    bootstrap_helper.print_images_in_statistics()

    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses were still saved in
    # the folderd (but not in the CSVs) for debug purpose. Let's remove them.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # Align CSVs with filtered images.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

if bootstrap:
    bootstrap(bootstrap_helper_train)
    bootstrap(bootstrap_helper_test)
