# realtime_tracking.py
import cv2
import dlib

def start_realtime_tracking():
    predictor_path = "data/models/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector(frame)
        for face in faces:
            landmarks = landmark_predictor(frame, face)
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

        cv2.imshow("Real-time Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_tracking()

