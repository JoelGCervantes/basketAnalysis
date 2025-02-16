from ultralytics import YOLO

model = YOLO('yolov10m')

results = model.predict('input_videos/RPReplay_Final1739697445.mp4', save=True)
print(results[0])

print('======== printing bx in boxes =========')
for box in results[0].boxes:
    print(box)


