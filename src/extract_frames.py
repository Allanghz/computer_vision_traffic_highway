import cv2
import os

# Path to video
video_dir = "C:\\Users\\obiwa\Documents\\projets\\data\\video\\"
output_dir = "C:\\Users\\obiwa\Documents\\projets\\data\\frames\\"

os.makedirs(output_dir, exist_ok=True)

for video_file in os.listdir(video_dir):
    if video_file.endswith(".avi"):
        video_path=os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_idx=0
        success,frame = cap.read()
        while success:
            frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            frame_idx +=1
            success,frame = cap.read()
        cap.release()

print("Extraction ended")