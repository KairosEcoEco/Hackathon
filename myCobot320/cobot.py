from pymycobot.mycobot import MyCobot
from ultralytics.utils.plotting import Annotator
import time
import cv2
from ultralytics import YOLO
import numpy as np
from queue import Queue
import threading

def angle_difference(angle1, angle2):
    # 각도를 0과 360도 사이로 정규화
    angle1 = angle1 % 360
    angle2 = angle2 % 360
    # 두 각도의 차이 계산
    diff = abs(angle1 - angle2)
    # 180도 이상의 차이가 나는 경우에는 보정
    if diff > 180:
        diff = 360 - diff
    return diff

def within_error_range(current_angles, target_angles, error_threshold):
    for cur_angle, tar_angle in zip(current_angles, target_angles):
        diff = angle_difference(cur_angle, tar_angle)
        if diff > error_threshold:
            return False
    return True

def robot_movement(q, mc):
    # Joint 초기각도 설정
    angles_0 = [0,0,0,0,0,0] 
    angles_1 = [80, 20, 30, 10, -90, 0] #
    angles_r = [-3, 60, 15, 20, -90, 80] #
    angles_p = [-22, 60, 15, 20, -90, 70] #
    angles_s = [13, 60, 15, 15, -90, 100] #
    angles_r_up = [-3, 0, 30, 0, -90, 90] #
    angles_p_up = [-22, 60, 15, 20, -90, 70] # 
    angles_s_up = [13, 0, 30, 0, -90, 90] #
    angles_b = [18, -10, -70, -10, 90, 0] #
    angles_b_down = [18, -20, -70, -20, 90, 0] #   
    delay = 2
    speed = 60
    # Gripper 초기값 및 잡았을 때 값 설정
    g_open = 100
    g_close = 15
    g_speed = 20
    g_delay =2
    # 로봇암 움직이기 시작
    mc.set_gripper_mode(0)
    mc.init_eletric_gripper()
    time.sleep(2)
    mc.send_angles(angles_0, speed)
    time.sleep(2)
    mc.send_angles(angles_1, speed)
    time.sleep(2)
    mc.set_gripper_value(g_open, g_speed)
    time.sleep(2)
    mc.set_color(0,255,0)
    print(f'시작, Joint: Angle1({angles_1}), Gripper: {g_open}')
    time.sleep(2)
    while True:
        command = q.get()
        rps = command[0]
        if rps == 'rock':
            print("주먹이 인식됐습니다.")
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_r, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_close, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_r_up, speed)
            time.sleep(delay)   
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_open, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_b_down, speed)
            time.sleep(delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            q.queue.clear()
        if rps == 'paper':
            print("보자기가 인식됐습니다.")
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_p, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_close, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_p_up, speed)
            time.sleep(delay)   
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_open, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_b_down, speed)
            time.sleep(delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            q.queue.clear()
            
        if rps == 'scissors':
            print("가위가 인식됐습니다.")
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_s, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_close, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_s_up, speed)
            time.sleep(delay)   
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            mc.set_gripper_value(g_open, g_speed)
            time.sleep(g_delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_b_down, speed)
            time.sleep(delay)
            mc.send_angles(angles_b, speed)
            time.sleep(delay)
            mc.send_angles(angles_0, speed)
            time.sleep(delay)
            mc.send_angles(angles_1, speed)
            time.sleep(delay)
            q.queue.clear()
            
       
def camera_thread(q, mc):
    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    confidence_threshold = 0.7
    cap.set(3, 640)  # 프레임의 너비, 가로 픽셀 수
    cap.set(4, 480)  # 프레임의 높이, 세로 필셀 수
    if cap.isOpened():
        print('카메라: ON')
    
    # YOLOv8 모델 로드
    model = YOLO('C:/Users/agshd/agv/3/best.pt')
    roi_x, roi_y, roi_w, roi_h = 100, 150, 400, 280  # 필요에 따라 조정
    while True:
        ret, frame = cap.read()
        if not ret:
            print('카메라에서 프레임을 읽을 수 없습니다.')
            break
        # 관심 영역(ROI) 설정
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # ROI를 연속적인 배열로 변환
        roi_contiguous = np.ascontiguousarray(roi)

        # YOLOv8 모델로부터 물체 감지
        results = model(roi_contiguous)
        annotator = Annotator(roi_contiguous, example=model.names)
        annotated_roi = results[0].plot()

        angles_1 = [80, 20, 30, 10, -90, 0]
        current_angles = mc.get_angles()
        error_threshold = 1.5 # 오차 허용범위
        if within_error_range(current_angles, angles_1, error_threshold):
            q.queue.clear()
            time.sleep(2)
            for result in results:
                if result.boxes.data.shape[0] > 0:
                    pred = result.boxes.data[0]
                    object_class = int(pred[5])
                    confidence = pred[4]
                    confidence_rate = confidence.item()
                    if confidence_rate > confidence_threshold:
                        x_center = (pred[0] + pred[2]) / 2
                        y_center = (pred[1] + pred[3]) / 2
                        x_center_value = round(x_center.item(),1)
                        y_center_value = round(y_center.item(),1)
                        if object_class == 0:
                            q.put(('paper', x_center_value, y_center_value))
                        elif object_class == 1:
                            q.put(('rock', x_center_value, y_center_value))
                        elif object_class == 2:
                            q.put(('scissors', x_center_value, y_center_value)) 
        frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = annotated_roi
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)  # ROI를 표시하는 녹색 사각형
         
        
        cv2.imshow("YOLOv8 추론", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
# 명령 큐 생성
    command_queue = Queue()
    mc = MyCobot('COM7', 115200)
# 로봇 움직임 및 카메라 쓰레드 시작
    robot_thread = threading.Thread(target=robot_movement, args=(command_queue,mc))
    camera_thread = threading.Thread(target=camera_thread, args=(command_queue,mc))
    robot_thread.start()
    camera_thread.start()

# 종료 시 쓰레드 정리
    robot_thread.join()
    camera_thread.join()