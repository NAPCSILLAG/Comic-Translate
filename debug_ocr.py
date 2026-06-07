import cv2
import numpy as np
from layout import LayoutDetector
from ocr import PPOCRv5Pipeline, OCRResult
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

def main():
    img = cv2.imread("input/teszt.jpg")
    if img is None:
        print("No image found!")
        sys.exit(1)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    layout = LayoutDetector()
    bubbles = layout.detect(img)
    print(f"YOLO detected {len(bubbles)} bubbles.")
    
    pipeline = PPOCRv5Pipeline()
    for i, b in enumerate(bubbles):
        print(f"\n--- Bubble {i} (bbox={b.bbox}) ---")
        try:
            res = pipeline.run(img_rgb, bbox=b.bbox, bubble_idx=i)
            print(f"Result count: {len(res)}")
            for r in res:
                print(f"  Text: '{r.text}' Conf: {r.confidence:.2f}")
        except Exception as e:
            print(f"Error on bubble {i}: {e}")

if __name__ == '__main__':
    main()
