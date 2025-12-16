import numpy as np
from PIL import Image
import math


def save_image_grid(images, path, rows=4):
    # images: (N, H, W, C), values uint8
    n = images.shape[0]
    cols = math.ceil(n / rows)
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx in range(n):
        img = images[idx]
        if c == 1:
            pil = Image.fromarray(img.squeeze(), mode='L')
            pil = pil.convert('RGB')
        else:
            pil = Image.fromarray(img)
        r = idx // cols
        col = idx % cols
        grid.paste(pil, (col * w, r * h))
    grid.save(path)


if __name__ == '__main__':
    print('visualizer ready')
