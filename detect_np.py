import cv2
import os
import shutil
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def get_image_files(input_folder):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def create_output_structure(input_folder, output_folder, image_file):
    relative_path = os.path.relpath(image_file, input_folder)
    output_image_path = os.path.join(output_folder, relative_path)
    output_image_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    return output_image_path

def process_image(image_file, input_folder, output_folder, predictor):
    image_path = image_file
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Failed to read image {image_path}")
        return False

    out_frame, out_frame_seg, out_frame_uv = predictor.predict(frame)
    
    # Write the processed image to the output folder
    output_image_path = create_output_structure(input_folder, output_folder, image_file)
    np.savez(output_image_path.split(".")[0] + ".npz", out_frame_seg)
    
    # cv2.imwrite(output_image_path, out_frame_seg)
    # cv2.imwrite(output_image_path.split(".")[0]+"_orig.png", out_frame)
    # cv2.imwrite(output_image_path.split(".")[0]+"_uv.png", out_frame_uv)

    return True

parser = ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, help="Set the input path to the folder containing images", default="frames"
)
parser.add_argument(
    "-o", "--output", type=str, help="Set the output path to the folder for saving processed images", default="detected"
)
args = parser.parse_args()

logger = GetLogger.logger(__name__)
predictor = Predictor()

# Get image files
input_folder = args.input
output_folder = args.output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = get_image_files(input_folder)
n_images = len(image_files)
logger.info(f"No of images {n_images}")

# Process images in parallel
done = 0

def update_progress(future):
    global done
    done += 1
    percent = int((done / n_images) * 100)
    sys.stdout.write(
        "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
    )
    sys.stdout.flush()

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_image, image_file, input_folder, output_folder, predictor) for image_file in image_files]
    for future in futures:
        future.add_done_callback(update_progress)

logger.info("Processing completed.")
