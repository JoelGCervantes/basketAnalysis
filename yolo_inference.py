from ultralytics import YOLO

# use yolov10m pretrained model for object detection on video and print some results
model = YOLO('yolov10m')

results = model.predict('input_videos/RPReplay_Final1739697445.mp4', save=True)
print(results[0])

print('======== printing bx in boxes =========')
for box in results[0].boxes:
    print(box)


