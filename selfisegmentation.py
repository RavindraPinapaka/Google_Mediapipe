import numpy as np
import time
import cv2 as cv
from mediapipe.python.solutions.selfie_segmentation import SelfieSegmentation


class SelfieSegment:
    def __init__(self, mod_sel=0):
        self.mod_sel = mod_sel                  # Model Selection
        self.selfie_seg = SelfieSegmentation(
            model_selection=mod_sel
        )    

    def selfie(self):
        ptime = 0
        source = cv.VideoCapture(0)
        bg_image = cv.imread("selfie.jpg")
        while source.isOpened():
            _, flip = source.read()
            frame = cv.resize(cv.flip(flip, 1), (800, 600))
            current_time = time.time()
            fps = str(int(1/(current_time-ptime)))
            ptime = current_time
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = self.selfie_seg.process(rgb)
            condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.5
            output_image = np.where(condition, frame, bg_image)
            cv.putText(output_image, "FPS: "+fps, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow("Webcam", output_image)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        source.release()
        cv.destroyAllWindows()

              
def main():
    selfie_seg = SelfieSegment()
    selfie_seg.selfie()


if __name__ == "__main__":
    main()
