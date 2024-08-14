import torch
import cv2
import dlib
import numpy as np

# 目のランドマークのIDを定義
LEFT_EYE_IDS = list(range(36, 42))
RIGHT_EYE_IDS = list(range(42, 48))
MOUTH_IDS = list(range(48, 60))

# ランドマークを画像に描画する関数
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    return image

# 目や口の動きをシミュレートする関数
def simulate_facial_movement(image, landmarks):
    left_eye = [landmarks[i] for i in LEFT_EYE_IDS]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDS]
    mouth = [landmarks[i] for i in MOUTH_IDS]

    for i in range(len(left_eye)):
        if i == 1 or i == 5:  
            left_eye[i] = (left_eye[i][0], left_eye[i][1] - 5)
            right_eye[i] = (right_eye[i][0], right_eye[i][1] - 5)
    
    for i in range(3, 6):  
        mouth[i] = (mouth[i][0], mouth[i][1] + 5)

    for (x, y) in left_eye + right_eye + mouth:
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)  
    
    return image

def align_landmarks(main_landmarks, ref_landmarks):
    ref_center = torch.mean(ref_landmarks, dim=0)
    main_center = torch.mean(main_landmarks, dim=0)
    
    ref_dist = torch.mean(torch.norm(ref_landmarks - ref_center, dim=1))
    main_dist = torch.mean(torch.norm(main_landmarks - main_center, dim=1))
    
    scale = ref_dist / main_dist
    
    aligned_landmarks = (main_landmarks - main_center) * scale + ref_center
    return aligned_landmarks

def calculate_displacement(aligned_landmarks, ref_landmarks):
    displacement = aligned_landmarks - ref_landmarks
    return displacement

def apply_displacement(ref_landmarks, displacement):
    updated_landmarks = ref_landmarks + displacement
    return updated_landmarks

def start_synchronized_tracking():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictor_path = "data/models/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)

    image = cv2.imread("scripts/inference/new_face.jpg")
    ref_faces = face_detector(image)
    if len(ref_faces) == 0:
        print("No faces found in the image.")
        return
    ref_landmarks = np.array([(p.x, p.y) for p in landmark_predictor(image, ref_faces[0]).parts()])
    ref_landmarks = torch.tensor(ref_landmarks, dtype=torch.float32).to(device)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector(frame)
        if len(faces) == 0:
            continue

        main_landmarks = np.array([(p.x, p.y) for p in landmark_predictor(frame, faces[0]).parts()])
        main_landmarks = torch.tensor(main_landmarks, dtype=torch.float32).to(device)

        aligned_landmarks = align_landmarks(main_landmarks, ref_landmarks)

        displacement = calculate_displacement(aligned_landmarks, ref_landmarks)

        updated_landmarks = apply_displacement(ref_landmarks, displacement).cpu().numpy()

        output_image = draw_landmarks(image.copy(), updated_landmarks)

        output_image = simulate_facial_movement(output_image, updated_landmarks)

        cv2.imshow("Image Landmarks Synced", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_synchronized_tracking()
