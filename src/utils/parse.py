import numpy as np
import gzip

def parse_label(file_path):
    with gzip.open(file_path) as f:
        f.read(4)
        f.read(4)
        labels = f.read()
        return np.frombuffer(labels, dtype=np.uint8)
    
def parse_image(file_path):
    with gzip.open(file_path) as f:
        f.read(4)
        image_count = int.from_bytes(f.read(4), "big")
        row_count = int.from_bytes(f.read(4), "big")
        col_count = int.from_bytes(f.read(4), "big")
        images = f.read()
        return np.frombuffer(images, dtype=np.uint8)\
                .reshape(image_count, row_count, col_count)
