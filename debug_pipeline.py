import cv2
from layout import ComicLayout
from ocr import get_ocr

def run_test():
    image_bgr = cv2.imread("input/teszt.jpg")
    if image_bgr is None:
        print("Nem talalhato az input/teszt.jpg")
        return

    layout = ComicLayout()
    bubbles = layout.detect(image_bgr)
    print(f"YOLO detected {len(bubbles)} bubbles")

    ocr = get_ocr()
    for i, b in enumerate(bubbles):
        bbox = b.get("bbox")
        results = ocr.extract(image_bgr, bbox=bbox, bubble_idx=i)
        texts = [r.text for r in results]
        print(f"Bubble {i} bbox {bbox}: OCR found {len(results)} items: {texts}")

if __name__ == "__main__":
    run_test()
