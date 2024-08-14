#!/bin/bash

# まず scripts ディレクトリを作成
if [ ! -d "scripts" ]; then
    mkdir "scripts"
    echo "Created directory: scripts"
else
    echo "Directory already exists: scripts"
fi

# サブディレクトリの構造を定義します
DIRECTORIES=(
    "data/raw_images"
    "data/landmarks"
    "data/models"
    "data/output"
    "scripts/preprocess"
    "scripts/training"
    "scripts/inference"
    "blender_models"
    "config"
    "logs"
)

# 各ディレクトリを作成します
for DIR in "${DIRECTORIES[@]}"; do
    if [ ! -d "$DIR" ];then
        mkdir -p "$DIR"
        echo "Created directory: $DIR"
    else
        echo "Directory already exists: $DIR"
    fi
done

# 各ファイルをそれぞれのディレクトリに移動します
mv apply_to_model.py "scripts/inference/" || echo "Failed to move apply_to_model.py"
mv face_synchronize.py "scripts/inference/" || echo "Failed to move face_synchronize.py"
mv new_realtime_tracking.py "scripts/inference/" || echo "Failed to move new_realtime_tracking.py"
mv realtime_tracking.py "scripts/inference/" || echo "Failed to move realtime_tracking.py"
mv preprocess_data.py "scripts/preprocess/" || echo "Failed to move preprocess_data.py"
mv landmark_detection.py "scripts/training/" || echo "Failed to move landmark_detection.py"
mv train_model.py "scripts/training/" || echo "Failed to move train_model.py"
mv main.py "./" || echo "Failed to move main.py"
mv README.md "./" || echo "Failed to move README.md"

# iniファイルの再生成
touch "config/project_config.ini"
touch "config/model_params.ini"

# logファイルの生成
touch "logs/execution.log"

echo "All files have been moved and necessary directories created."
