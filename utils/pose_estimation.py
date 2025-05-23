
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_landmarks(image):
    """Returns landmarks detected by MediaPipe Pose model."""
    results = pose.process(image)
    return results.pose_landmarks.landmark if results.pose_landmarks else None
