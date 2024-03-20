import cv2
import numpy as np
from ultralytics import YOLO

def detect_edges(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for yellow color
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([18, 255, 255])

    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply Canny edge detection
    edges = cv2.Canny(mask, 70, 190)
    return edges

def threshold_binary_image(img):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Thresholding to create binary image
    binary_img = np.zeros_like(blurred)
    binary_img[(blurred > 100) & (blurred <= 255)] = 255
    return binary_img

def detect_lines(edges):
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    return lines

def detect_object(frame):
    # Use YOLO model to detect objects
    model = YOLO("yolov8m.pt")
    results = model.predict(frame)
    return results[0]


def annotate_frame(frame, cords, label_with_info):
    # Draw bounding box and label on the frame
    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
    cv2.putText(frame, label_with_info, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def process_frame(frame):
    # Detect edges and lines
    edges = detect_edges(frame)
    lines = detect_lines(edges)

    # Draw detected lines on the frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Detect object using YOLO
    result = detect_object(frame)

    # Process detected objects
    objects_info = []
    column1_start = (frame.shape[1] // 3, 0)
    column1_end = (frame.shape[1] // 3, frame.shape[0])
    column2_start = (2 * frame.shape[1] // 3, 0)
    column2_end = (2 * frame.shape[1] // 3, frame.shape[0])
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()


        # Calculate centroid
        centroid_x = (cords[0] + cords[2]) / 2
        centroid_y = (cords[1] + cords[3]) / 2
        # Draw three columns
        
        
        cv2.line(frame, column1_start, column1_end, (255, 0, 0), 2)
        cv2.line(frame, column2_start, column2_end, (255, 0, 0), 2)
        print([column1_start, column1_end],[column2_start, column2_end])
        print(column2_start)
        if centroid_x < column1_start[0]:
            direction = "left"
        elif centroid_x > column1_start[0] and centroid_x < column2_start[0] :
            direction = "centre"
        else:
            direction = "right"

        # Add information to the label
        label_with_info = f"{label} {direction}"
        objects_info.append([label_with_info, (centroid_x, centroid_y), prob])
        print(objects_info)
        if prob > 0.75:
        # Annotate frame
            annotate_frame(frame, cords, label_with_info)

    return frame, objects_info

if __name__ == "__main__":
    # Load the image
    frame = cv2.imread("IMG20240313094126.jpg")

    # Process the frame
    annotated_frame, _ = process_frame(frame)
    # Display the annotated image
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
