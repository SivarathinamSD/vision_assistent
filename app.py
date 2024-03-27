import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

engine = pyttsx3.init()

def detect_object(frame):
    model = YOLO("yolov8m.pt")
    results = model.predict(frame)
    return results[0]

def annotate_frame(frame, cords, label_with_info):
    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
    cv2.putText(frame, label_with_info, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def process_frame(frame):
    result = detect_object(frame)
    objects_info = []
    column1_start = (frame.shape[1] // 3, 0)
    column1_end = (frame.shape[1] // 3, frame.shape[0])
    column2_start = (2 * frame.shape[1] // 3, 0)
    column2_end = (2 * frame.shape[1] // 3, frame.shape[0])

    cv2.line(frame, column1_start, column1_end, (255, 0, 0), 2)
    cv2.line(frame, column2_start, column2_end, (255, 0, 0), 2)
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()

        centroid_x = (cords[0] + cords[2]) / 2
        centroid_y = (cords[1] + cords[3]) / 2
        
        direction = determine_direction(centroid_x, column1_start, column2_start)

        label_with_info = f"{label} ({direction})"
        objects_info.append([label_with_info, (centroid_x, centroid_y), prob])
        if prob > 0.75:
            annotate_frame(frame, cords, label_with_info)

    return frame, objects_info

def determine_direction(centroid_x, column1_start, column2_start):
    if centroid_x < column1_start[0]:
        return "Left"
    elif centroid_x > column1_start[0] and centroid_x < column2_start[0]:
        return "Center"
    else:
        return "Right"

def edge_detection(frame, threshold1, threshold2):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray_frame, threshold1, threshold2)
    return mask

def lane_detection(mask):
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    return left_base, right_base

def perspective_transformation(frame, pts1, pts2):
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
    return transformed_frame

def draw_circles(frame, points):
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def show_frames(transformed_frame, annotated_frame, msk):
    cv2.imshow("Birds Eye", transformed_frame)
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.imshow("Detected", msk)

def navigate(detected_objects):
    objects_left = []
    objects_right = []
    for obj in detected_objects:
        if "left" in obj.lower():
            objects_left.append(obj)
        elif "right" in obj.lower():
            objects_right.append(obj)
    
    if objects_left:
        engine.say("Object(s) detected on the left. Turn right to avoid.")
        engine.runAndWait()
        print("Object(s) detected on the left:", objects_left)
        print("Turn right to avoid.")
    elif objects_right:
        engine.say("Object(s) detected on the right. Turn left to avoid.")
        engine.runAndWait()
        print("Object(s) detected on the right:", objects_right)
        print("Turn left to avoid.")
    else:
        engine.say("No objects detected in the way.")
        engine.runAndWait()
        print("No objects detected in the way.")

def nothing(x):
    pass

if __name__ == "__main__":
    vidcap = cv2.VideoCapture(0)
    success, image = vidcap.read()
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Threshold 1", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("Threshold 2", "Trackbars", 200, 255, nothing)

    while True:
        success, image = vidcap.read()
        frame = cv2.resize(image, (640, 480))
        
        tl = (222, 387)
        bl = (70, 472)
        tr = (400, 380)
        br = (538, 472)
        
        draw_circles(frame, [tl, bl, tr, br])
        
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        transformed_frame = perspective_transformation(frame, pts1, pts2)

        if transformed_frame.shape[0] < 240:
            threshold1 = 50
            threshold2 = 100

            left_base, right_base = lane_detection(transformed_frame)
            if left_base < 320:
                navigate(["Left"])
            elif right_base > 320:
                navigate(["Right"])
        else:
            annotated_frame, _ = process_frame(frame)

            show_frames(transformed_frame, annotated_frame, np.zeros_like(transformed_frame))

            detected_objects = [obj[0] for obj in process_frame(frame)[1]]
            for obj in detected_objects:
                print(obj)

            navigate(detected_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
