import laspy
import numpy as np
from pathlib import Path
from sklearn.metrics import jaccard_score

if __name__ == "__main__":
    cloud_dir = "./test/France/clouds"
    num_classes = 5

    cloud_names = list(Path(cloud_dir).glob("*.[lL][aS][zZsS]"))
    all_preds = []
    all_targets = []

    for cloud_name in cloud_names:
        test = laspy.read(cloud_name)

        preds = np.uint8(test.prediction)
        targets = np.int8(test.classification)

        all_preds.extend(preds)
        all_targets.extend(targets)

    IoUs = jaccard_score(all_targets, all_preds, labels=[1, 2, 3, 4], average=None)
    print(IoUs)
