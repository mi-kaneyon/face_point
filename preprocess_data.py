import cv2
import os
import time

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_images(output_dir, capture_duration=20, capture_limit=None):  # 20秒に設定、枚数制限なし
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    count = 0

    ensure_directory_exists(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_name = f"image_{count}.jpg"
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, frame)
        print(f"Image saved: {output_path}")

        count += 1

        # 指定時間が経過したら終了
        if time.time() - start_time > capture_duration:
            break

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 20秒間のデータキャプチャを開始
    capture_images("data/raw_images", capture_duration=20)  # 20秒間に変更
