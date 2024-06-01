import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    ssim_value, _ = ssim(img1, img2, full=True,data_range=1.0)
    return ssim_value

# Function to process video frames
def process_video(original_video_path, compressed_video_path):
    # Open the videos
    original_video = cv2.VideoCapture(original_video_path)
    compressed_video = cv2.VideoCapture(compressed_video_path)

    # Check if videos are opened successfully
    if not original_video.isOpened():
        print(f"Error opening original video: {original_video_path}")
        return [], []

    if not compressed_video.isOpened():
        print(f"Error opening compressed video: {compressed_video_path}")
        return [], []

    # Initialize lists to store PSNR and SSIM values
    psnr_values = []
    ssim_values = []
    current_frame = 0
    # Read frames from videos
    while True:
        ret_original, frame_original = original_video.read()
        ret_compressed, frame_compressed = compressed_video.read()

        # Break the loop if we have reached the end of the videos
        if not ret_original or not ret_compressed:
            break

        # Convert frames to grayscale if they are not already
        if len(frame_original.shape) == 3:
            frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        if len(frame_compressed.shape) == 3:
            frame_compressed = cv2.cvtColor(frame_compressed, cv2.COLOR_BGR2GRAY)

        total_frames = int(
            min(original_video.get(cv2.CAP_PROP_FRAME_COUNT), compressed_video.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Calculate PSNR and SSIM
        psnr = calculate_psnr(frame_original, frame_compressed)
        ssim_value = calculate_ssim(frame_original, frame_compressed)

        # Append values to lists
        psnr_values.append(psnr)
        ssim_values.append(ssim_value)
        current_frame += 1
        print(f"Processed {current_frame}/{total_frames} frames")

    # Release the videos
    original_video.release()
    compressed_video.release()

    return psnr_values, ssim_values

base_dir = r'C:\Users\leont\PycharmProjects\dimplomska\kodeki\\'
codec_dirs = ['264', '265']
video_names = ['BigBuckBunny', 'EarlyRide', 'Horror']

for video_name in video_names:
    for codec_dir in codec_dirs:
        # Construct the path for the original video
        original_video_path = os.path.join(base_dir, 'mp4', video_name + '.mp4')

        # Construct the path for the compressed video based on the codec directory
        compressed_video_path = os.path.join(base_dir, codec_dir, video_name + '.' + codec_dir)

        # Ensure the compressed video path ends with the correct extension
        print(f"Comparing: {compressed_video_path} \n {original_video_path}")

        psnr_values, ssim_values = process_video(original_video_path, compressed_video_path)

        if psnr_values and ssim_values:  # Only plot if there are valid values
            # Plot PSNR and SSIM values for each comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(psnr_values)
            plt.title(f'PSNR Values for {video_name} in {codec_dir}')
            plt.xlabel('Frame Number')
            plt.ylabel('PSNR')

            plt.subplot(1, 2, 2)
            plt.plot(ssim_values)
            plt.title(f'SSIM Values for {video_name} in {codec_dir}')
            plt.xlabel('Frame Number')
            plt.ylabel('SSIM')

            plt.tight_layout()
            plt.show()
