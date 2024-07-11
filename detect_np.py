import cv2
import os
import shutil
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch

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

def rescale_image(image, max_dim=512):
    height, width = image.shape[:2]
    if max(height, width) > max_dim:
        scaling_factor = max_dim / max(height, width)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def process_image(image_file, input_folder, output_folder, predictor, rescale):
    image_path = image_file
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Failed to read image {image_path}")
        return False

    # Rescale the image if necessary
    if rescale:
        frame = rescale_image(frame)

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
parser.add_argument(
    "-r", "--rescale", action='store_true', help="Rescale images if their maximum dimension is larger than 512 pixels", default=False
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

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info("Running on GPU")
else:
    logger.info("Running on CPU")

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

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_image, image_file, input_folder, output_folder, predictor, args.rescale) for image_file in image_files]
    for future in futures:
        future.add_done_callback(update_progress)

logger.info("Processing completed.")
