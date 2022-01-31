import cv2 as cv    # Importing Libraries and Constants
import time
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks


class PoseDetector:  # Class and Constructor

    def __init__(self):
        self.pose = Pose()

    def pose_detector(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h = frame.shape[0]  # or use this method: h,w,c = frame.shape
        w = frame.shape[1]
        if results.pose_landmarks:
            draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)
        # This is only for Blue dots Purpose
        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x*w), int(lm.y*h)
            cv.circle(frame, (x, y), 4, (255, 0, 0), cv.FILLED)


def main():  # Main function creating
    source = cv.VideoCapture(0)  # Read input, you can use video file as input instead of 0
    ptime = 0
    detector = PoseDetector()  # Object Creating
    while source.isOpened:
        _, flip = source.read()
        frame = cv.resize(cv.flip(flip, 1), (800, 600))
        current_time = time.time()
        fps = str(int(1 / (current_time - ptime)))
        ptime = current_time
        detector.pose_detector(frame)   # Method calling
        cv.putText(frame, "FPS: " + fps, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.imshow("Webcam", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    source.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
