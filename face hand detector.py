import cv2 as cv  # Libraries and Constants
import time
from mediapipe.python.solutions.drawing_utils import *
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.face_mesh_connections import *
from scipy.spatial import distance


class Mpipe:  # Class and Constructor
    """
    This is a class for Hand Detection and Face Detection           

    Attributes:
    -----------
    mode (boolean): 
        If set to False, the solution treats the input images as a video stream.
    max_hands (int):
        Maximum number of hands to detect. Default to 2.
    detection_con (float):
        Minimum detection confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered
         successful. Default to 0.5.
    trac_con (float):
        Minimum tracking confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be
        considered tracked successfully, or otherwise hand detection will be invoked automatically on the next
        input image.
    mod_com (int):
       Model complexity of the hand landmark model 0 or 1. Landmark accuracy as well as inference latency generally
       go up with the model complexity. Default to 1.
    mod_sel (int):
        An integer index 0 or 1. Use 0 to select a short-range model that works best for faces within 2 meters from the
         camera, and 1 for a full-range model best for faces within 5 meters.
    max_faces (int):
        Maximum number of faces to face mesh. Default to 1
    ref_lm (boolean):
        Whether to further refine the landmark coordinates around the eyes and lips, and output additional landmarks
        around the irises by applying the Attention Mesh Model. Default to false.
    
    Methods:
    --------
    result():
        It returns hand process results
    hand_landmarks():
        It draws landmarks on the hands
    cn_hands():
        It prints the number of hands detected
    hand_bbox():
        It draws the bounding box of hands
    hand_type_score():
        It prints the hand type ('i.e' is it a Right or Left) label and score
    distance():
        It prints the distance between both hand center points
    face_detection():
        It detects the face with bounding box
    face_mesh():
        It draws the face landmarks and line between landmarks
    
    Suggestion:
    -----------
    Set detection_con, trac_con 0.5 to 0.7 for less false positives
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, trac_con=0.5, mod_com=0, mod_sel=1, max_faces=1,
                 ref_lm=False):
        """
        Constructs all the necessary attributes for the Mpipe object.
        __init__ accepts the parameter values from user
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.trac_con = trac_con
        self.mod_com = mod_com
        self.hands = Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            model_complexity=mod_com,
            min_detection_confidence=detection_con,
            min_tracking_confidence=trac_con)
        self.mod_sel = mod_sel
        self.face = FaceDetection(
            model_selection=mod_sel, 
            min_detection_confidence=detection_con
        )
        self.max_faces = max_faces
        self.ref_lm = ref_lm
        self.mesh = FaceMesh(
            static_image_mode=mode,
            refine_landmarks=ref_lm,
            max_num_faces=max_faces,
            min_detection_confidence=detection_con,
            min_tracking_confidence=trac_con
        )        
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.rgb = cv.COLOR_BGR2RGB
        self.process = self.hands.process  # This is only for improve to  Performance

        # Methods

    def result(self, frame):
        """
        It returns hand process results
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        rgb = cv2.cvtColor(frame, cv.COLOR_BGR2RGB)
        process = self.process(rgb)
        return process

    def hand_landmarks(self, frame):
        """
        It draws landmarks on the hands
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        result = self.result(frame)
        mxhands = str(self.max_hands)
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                draw_landmarks(frame, hand_lms, HAND_CONNECTIONS)
        cv.putText(frame, "MaxHands: " + mxhands, (95, 30), self.font, 0.7, (255, 0, 255), 2)

    def cn_hands(self, frame):
        """
        It prints the number of hands detected
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        result = self.result(frame)
        if result.multi_hand_landmarks:
            count = str(len(result.multi_hand_landmarks))
            cv.putText(frame, "DetectedHands: " + count, (260, 30), self.font, 0.7, (0, 0, 255), 2)
        else:
            cv.putText(frame, "DetectedHands: 0", (260, 30), self.font, 0.7, (0, 0, 0), 2)

    def hand_bbox(self, frame):
        """
        It draws the bounding box of hands
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        result = self.result(frame)
        h, w, _ = frame.shape
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

    def hand_type_score(self, frame):
        """
        It prints the hand type(i.e. is it a Right or Left) label and score
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        result = self.result(frame)
        h, w, _ = frame.shape
        if result.multi_hand_landmarks:
            for classification, hand_lms in zip(result.multi_handedness, result.multi_hand_landmarks):
                hands = {}
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                hands["type"] = classification.classification[0].label
                cv.putText(frame, hands["type"], (x_min - 30, y_min - 30), self.font, 0.7, (255, 0, 255), 2)
                hands["score"] = str(round(classification.classification[0].score * 100, 2))
                cv.putText(frame, "Confidence: " + hands["score"] + "%", (x_min - 50, y_max + 50), self.font, 0.5,
                           (0, 255, 0), 2)

    def distance(self, frame):
        """
        It prints the distance between both hand center points
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        result = self.result(frame)
        h, w, _ = frame.shape
        pt1 = []
        pt2 = []
        if result.multi_hand_landmarks:
            if len(result.multi_hand_landmarks) == 2:
                for hand_num, hand_lms in enumerate(result.multi_hand_landmarks):
                    for lm_id, lm in enumerate(hand_lms.landmark):
                        if hand_num == 0 and lm_id == 9:
                            x1, y1 = int(lm.x * w), int(lm.y * h)
                            pt1.append([x1, y1])
                            cv.circle(frame, (x1, y1), 8, (255, 0, 255), cv.FILLED)
                    for lm_id, lm in enumerate(hand_lms.landmark):
                        if hand_num == 1 and lm_id == 9:
                            x2, y2 = int(lm.x * w), int(lm.y * h)
                            pt2.append([x2, y2])
                            cv.circle(frame, (x2, y2), 8, (255, 0, 255), cv.FILLED)
                cv.line(frame, (pt1[0][0], pt1[0][1]), (pt2[0][0], pt2[0][1]), (255, 0, 255), 3)
                dist = round(distance.euclidean(pt1, pt2) / 10, 2)
                cv.putText(frame, f'Distance: {dist} CM', (470, 30), self.font, 0.7, (255, 0, 255), 2)

    def face_detection(self, frame):
        """
        It detects the face with bounding box
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        h = frame.shape[0]  # or use this method: h, w, _ = frame.shape
        w = frame.shape[1]
        rgb = cv.cvtColor(frame, self.rgb)
        result = self.face.process(rgb)
        if result.detections:
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int(box.width * w)
                y2 = int(box.height * h)
                score = str(round(detection.score[0] * 100, 2))
                cv.rectangle(frame, (x1 - 10, y1 - 50, x2 + 10, y2 + 50), (0, 255, 0), 2)
                cv.putText(frame, "Confidence: " + score + "%", (x1, y1 - 70), self.font, 0.6, (255, 0, 255), 2)

    def face_mesh(self, frame):
        """
        It draws the face landmarks and line between landmarks
        Parameters:
        ----------
        frame (image):
        It reads the input image
        """
        rgb = cv.cvtColor(frame, self.rgb)
        result = self.mesh.process(rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                draw_spec = DrawingSpec(color=(0, 0, 255), thickness=0, circle_radius=2)
                draw_landmarks(frame, face_landmarks, FACEMESH_CONTOURS, draw_spec)


def main():  # Main function
    source = cv.VideoCapture(0)  # Read input
    ptime = 0
    pts = np.array([[[10, 10], [790, 10], [790, 40], [10, 40]]])
    detector = Mpipe()  # Object Creating
    while source.isOpened:
        _, flip = source.read()
        frame = cv.resize(cv.flip(flip, 1), (800, 600))
        cv.fillPoly(frame, pts=pts, color=(255, 255, 255))
        current_time = time.time()
        fps = str(int(1 / (current_time - ptime)))
        ptime = current_time
        detector.hand_landmarks(frame)  # Methods Calling
        detector.cn_hands(frame)
        detector.hand_bbox(frame)
        detector.hand_type_score(frame)
        detector.distance(frame)
        detector.face_detection(frame)
        detector.face_mesh(frame)
        cv.putText(frame, "FPS: " + fps, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.imshow("Webcam", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    source.release()
    cv.destroyAllWindows()


if __name__ == "__main__":  # Special Variable
    main()
