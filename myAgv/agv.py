import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading
import time
from queue import Queue 

agv = MyAgv("/dev/ttyAMA2", 115200)
exit_flag = threading.Event()
def process_frame(frame):
    height, width, _ = frame.shape 
    roi_height = int(height / 5) 
    roi_top = height - roi_height 
    roi = frame[roi_top:, :] 
    non_roi = frame[:roi_top, :]

    # Draw ROI rectangle
    cv2.rectangle(roi, (0, 0), (width, roi_height), (255, 255, 0), 2) 
    # Draw center line
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 2) 

    # Convert to HSV for color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_non_roi = cv2.cvtColor(non_roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for white and red
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)
    # Define color ranges for green
    lower_green = np.array([35, 50, 50], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    lower_yellow = np.array([15, 150, 20], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    # Create masks for colors
    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_non_roi, lower_green, upper_green)
    white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    # Check for red outside the ROI
    if cv2.countNonZero(green_mask) > 50000:
        return "STOP"
    # Check for yellow inside the ROI
    if cv2.countNonZero(yellow_mask) > 10000:
        return "YELLOW"
    # Process white mask to find contours
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #for contour in contours:
    #    area = cv2.contourArea(contour)
    #    if area > 500:  # Adjust the area threshold based on your specific needs
    #        return "CROSSWALK"
                  
    # If multiple white lines are detected within the ROI, assume a crosswalk
    # Adjust the threshold based on the expected number of lines in a crosswalk  
    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if cx > 470:
                return "LEFT"
            elif cx < 190:
                return "RIGHT"
            else:
                return "GO"
        else:
            return "STOP"
    return None

def control_thread(frame_queue):
    while not exit_flag.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            result = process_frame(frame)
            if result == "YELLOW":
                agv.go_ahead(1)
                time.sleep(0.5)
                agv.stop()
            #elif result == "CROSSWALK":
            #    agv.stop()
            #    time.sleep(5)
            #    agv.go_ahead(1)
            #    time.sleep(0.8)
            elif result == "LEFT":
                print("LEFT")
                agv.clockwise_rotation(1)
                time.sleep(0.02)
                agv.go_ahead(1)
                time.sleep(0.01)
                agv.stop()
            elif result == "RIGHT":
                print("RIGHT")
                agv.counterclockwise_rotation(1)
                time.sleep(0.02)
                agv.go_ahead(1)
                time.sleep(0.01)
                agv.stop()
            elif result == "GO":
                print("GO")
                agv.go_ahead(1)
                time.sleep(0.2)
            elif result == "STOP":
                print("STOP")
                agv.stop()

def camera_thread(frame_queue):
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("camera error")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
            frame_s = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA) 
            cv2.imshow("ORG", frame_s) 
            if cv2.waitKey(25) & 0xFF == ord('q'):
                exit_flag.set()
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = Queue(maxsize=1)
    camera_thread = threading.Thread(target=camera_thread, args=(frame_queue,))
    camera_thread.start()
    control_thread = threading.Thread(target=control_thread, args=(frame_queue,))
    control_thread.start()
    camera_thread.join()
    control_thread.join()
    
