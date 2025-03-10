from ultralytics.data.converter import convert_coco
import os

# Define your dataset root
dataset_root = '/storage/ice1/0/2/yzhang3942/llm-guided-evolution/datasets/coco'

# Convert training annotations using the instances file
convert_coco(
    labels_dir=os.path.join(dataset_root, "annotations", "instances_train2017.json"),
    save_dir=os.path.join(dataset_root, "train2017"),
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False
)

# And similarly for validation annotations
convert_coco(
    labels_dir=os.path.join(dataset_root, "annotations", "instances_val2017.json"),
    save_dir=os.path.join(dataset_root, "val2017"),
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False
)
