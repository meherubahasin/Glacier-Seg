import random
import numpy as np
import cv2

def normalize(arr):
    arr = arr.astype(np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-5)


def augment_patch(img, label):
    if random.random() < 0.5:
        img = np.flip(img, axis=0)
        label = np.flip(label, axis=0)
    if random.random() < 0.5:
        img = np.flip(img, axis=1)
        label = np.flip(label, axis=1)

    if random.random() < 0.5:
          # Random brightness adjustment
          factor = 1.0 + (0.2 * (random.random() - 0.5))  # ±10%
          img = np.clip(img * factor, 0, 255).astype(np.uint8)
  
      if random.random() < 0.5:
          # Random contrast adjustment
          alpha = 1.0 + (0.3 * (random.random() - 0.5))  # ±15%
          img = np.clip(128 + alpha * (img - 128), 0, 255).astype(np.uint8)
  
      if random.random() < 0.5:
          # Gaussian noise
          noise = np.random.normal(0, 5, img.shape)
          img = np.clip(img + noise, 0, 255).astype(np.uint8)
  
      if random.random() < 0.5:
          # Gaussian blur
          ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img, label


