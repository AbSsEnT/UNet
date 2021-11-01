import os

import pandas as pd

INP_DATA_PATH = os.path.join(".", "data", "train_images")
OUT_DATA_PATH = os.path.join(".", "data", "train_labels")

if __name__ == "__main__":
    images_filenames = sorted(os.listdir(INP_DATA_PATH))
    masks_filenames = sorted(os.listdir(OUT_DATA_PATH))

    annotations_df = pd.DataFrame(data={"images_filenames": images_filenames,
                                        "masks_filenames": masks_filenames})
    annotations_df.to_csv(os.path.join(".", "data", "annotations.csv"), index=False)
