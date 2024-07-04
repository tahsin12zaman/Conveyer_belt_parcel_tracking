import cv2
import torch
import time
import os

# Load YOLOv5 model (assuming similar structure as YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# Function to detect parcels
def detect_parcels(frame):
    results = model(frame)
    parcels = []
    for result in results.xyxy[0]:
        if result[5] == 0:  # Assuming parcel class ID is 0, adjust based on your model's class IDs
            x1, y1, x2, y2 = map(int, result[:4])
            parcels.append((x1, y1, x2, y2))
    return parcels


# Function to track parcels
def track_parcels(frame, trackers):
    new_trackers = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            new_trackers.append(tracker)
    return new_trackers


# Function to handle power outage
def handle_power_outage(trackers, detected_parcels):
    existing_parcels = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            existing_parcels.append(bbox)

    new_parcels = []
    for parcel in detected_parcels:
        is_new = True
        for existing in existing_parcels:
            iou = calculate_iou(parcel, existing)
            if iou > 0.5:  # IOU threshold to determine if it's the same parcel
                is_new = False
                break
        if is_new:
            new_parcels.append(parcel)

    return new_parcels


# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Main function to count parcels
def count_parcels(video_path):
    if not os.path.exists(video_path):
        print(f"Error: The video path {video_path} does not exist.")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0

    trackers = []
    parcel_count = 0
    power_outage = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display frame for debugging
        cv2.imshow('Debug Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if power_outage:
            detected_parcels = detect_parcels(frame)
            new_parcels = handle_power_outage(trackers, detected_parcels)
            parcel_count += len(new_parcels)
            power_outage = False  # Reset after handling
        else:
            trackers = track_parcels(frame, trackers)
            if len(trackers) == 0:
                detected_parcels = detect_parcels(frame)
                for parcel in detected_parcels:
                    try:
                        tracker = cv2.TrackerKCF_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerKCF_create()
                    tracker.init(frame, tuple(parcel))
                    trackers.append(tracker)
                parcel_count += len(detected_parcels)

        # Simulate power outage
        if time.time() % 30 < 1:  # Example condition to simulate power outage
            power_outage = True
            time.sleep(10)  # Simulate the 10-minute power outage
            continue

        # Display the frame with parcel count
        cv2.putText(frame, f'Parcel Count: {parcel_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return parcel_count


# Example usage
video_path = 'These_conveyor_belts_are_a_trip.mp4'  # Replace with your actual video path
total_parcels = count_parcels(video_path)
print(f'Total parcels counted: {total_parcels}')
