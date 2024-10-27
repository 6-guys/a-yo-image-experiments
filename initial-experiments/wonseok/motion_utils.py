# motion_utils.py

import cv2
import numpy as np

def calculate_motion_vectors(frames):
    """
    Calculate motion vectors between consecutive frames.
    
    Parameters:
    frames (numpy array): Array of frames (shape: num_frames, height, width, channels).
    
    Returns:
    motion_vectors (numpy array): Array of motion vectors (shape: num_frames-1, height, width, 2).
    """
    motion_vectors = []
    
    # Convert frames to uint8 if needed, assuming the input is in [0, 1] range
    if frames.dtype == np.float64 or frames.dtype == np.float32:
        frames_uint8 = (frames * 255).astype(np.uint8)
    else:
        frames_uint8 = frames  # If frames are already in uint8, no conversion needed
    
    # Convert frames to grayscale for optical flow calculation
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_uint8]
    
    # Loop through frames and calculate motion vectors between consecutive frames
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        motion_vectors.append(flow)
    
    return np.array(motion_vectors)


def combine_frames_and_motion_vectors(frames, motion_vectors):
    """
    Combine image frames and motion vectors along the channel axis.
    
    Parameters:
    frames (numpy array): Array of frames (shape: num_frames, height, width, 3).
    motion_vectors (numpy array): Array of motion vectors (shape: num_frames-1, height, width, 2).
    
    Returns:
    combined_data (numpy array): Array of combined frames and motion vectors (shape: num_frames-1, height, width, 5).
    """
    combined_data = []
    
    # Combine each frame with its corresponding motion vector
    for i in range(len(motion_vectors)):
        combined = np.concatenate((frames[i+1], motion_vectors[i]), axis=-1)
        combined_data.append(combined)
    
    return np.array(combined_data)
