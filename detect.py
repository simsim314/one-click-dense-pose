import cv2
import os
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys


def get_image_files(input_folder):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]


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

# Process each image
done = 0
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning(f"Failed to read image {image_path}")
        continue

    out_frame, out_frame_seg, out_frame_uv = predictor.predict(frame)
    
    # Write the processed image to the output folder
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, out_frame_seg)
    cv2.imwrite(output_image_path.split(".")[0]+"_orig.png", out_frame)
    cv2.imwrite(output_image_path.split(".")[0]+"_uv.png", out_frame_uv)

    done += 1
    percent = int((done / n_images) * 100)
    sys.stdout.write(
        "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
    )
    sys.stdout.flush()

logger.info("Processing completed.")
