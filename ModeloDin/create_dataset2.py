import os
import cv2
import numpy as np
import mediapipe as mp

# Define the path to the data_imgs folder
data_folder = "data_imgs"
output_folder = "output_data"  # Create this folder manually before running the script

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Iterate over all folders in the data_imgs folder
for sign_folder in os.listdir(data_folder):
    sign_path = os.path.join(data_folder, sign_folder)
    if os.path.isdir(sign_path):
        # Create a folder in the output directory for each sign
        output_sign_folder = os.path.join(output_folder, sign_folder)
        os.makedirs(output_sign_folder, exist_ok=True)

        # Iterate over all sets in the current sign folder
        for set_folder in os.listdir(sign_path):
            set_path = os.path.join(sign_path, set_folder)
            if os.path.isdir(set_path):
                # Create an empty list to store keypoints for this set
                set_keypoints = []

                # Iterate over all images in the current set folder
                for filename in os.listdir(set_path):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        # Read the image
                        image = cv2.imread(os.path.join(set_path, filename))

                        # Convert the image to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Perform hand keypoint detection using mediapipe
                        results = hands.process(image_rgb)
                        if results.multi_hand_landmarks:
                            hand_keypoints = results.multi_hand_landmarks[0].landmark

                            # Flatten the keypoints and append to the list
                            keypoints_array = np.array([[point.x, point.y, point.z] for point in hand_keypoints]).flatten()
                            set_keypoints.append(keypoints_array)

                # Convert the keypoints list for this set to a numpy array
                set_keypoints_array = np.array(set_keypoints)

                # Save the keypoints array as a numpy file for this set
                output_filename = os.path.join(output_sign_folder, f"{set_folder}_keypoints.npy")
                np.save(output_filename, set_keypoints_array)
