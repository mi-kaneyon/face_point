# apply_to_model.py
import bpy

def apply_landmarks_to_model(landmarks):
    # この関数でBlenderモデルにランドマークを適用する
    print(f"Applying landmarks to model: {landmarks}")
    # BlenderのPython APIを使ってモデルのボーンを動かす処理をここに追加

if __name__ == "__main__":
    # 例として、固定されたランドマークデータを適用
    dummy_landmarks = [(30, 40), (50, 60), (70, 80)]
    apply_landmarks_to_model(dummy_landmarks)

