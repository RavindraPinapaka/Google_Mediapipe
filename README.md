# mediapipe_face_hand_detector
# this is a face and hand detector module written in python with opencv and google mediapipe
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
