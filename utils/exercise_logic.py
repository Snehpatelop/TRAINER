
import numpy as np

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def detect_squat(hip, knee, ankle):
    """Detects a squat based on knee angle."""
    angle = calculate_angle(hip, knee, ankle)
    return angle < 90  # If knee angle is less than 90, squat detected
