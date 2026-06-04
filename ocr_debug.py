# ocr_debug.py

import os
import json
import cv2
from datetime import datetime


class OCRDebug:

    def __init__(self, out_dir="debug"):
        self.out_dir = out_dir
        self.crop_dir = os.path.join(out_dir, "crops")

        os.makedirs(self.crop_dir, exist_ok=True)

        self.records = []

    def save_crop(
        self,
        image,
        bubble_id,
        bbox,
        crop,
        text="",
        engine="NONE",
        status="unknown",
        confidence=0.0
    ):

        x1, y1, x2, y2 = bbox

        crop_file = f"bubble_{bubble_id:03d}.png"
        crop_path = os.path.join(self.crop_dir, crop_file)

        if crop is not None and crop.size > 0:
            cv2.imwrite(crop_path, crop)
        else:
            crop_file = None

        self.records.append({
            "bubble_id": bubble_id,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "engine": engine,
            "status": status,
            "confidence": float(confidence),
            "text": text,
            "crop": crop_file
        })

    def save_report(self):

        report_path = os.path.join(
            self.out_dir,
            "ocr_report.json"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                self.records,
                f,
                ensure_ascii=False,
                indent=2
            )

    def save_overlay(self, image):

        overlay = image.copy()

        for rec in self.records:

            x1, y1, x2, y2 = rec["bbox"]

            status = rec["status"]

            if status == "ok":
                color = (0, 255, 0)

            elif status == "empty":
                color = (0, 255, 255)

            else:
                color = (0, 0, 255)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                overlay,
                f'#{rec["bubble_id"]}',
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        cv2.imwrite(
            os.path.join(self.out_dir, "overlay.png"),
            overlay
        )

    def print_summary(self):

        total = len(self.records)

        ok = sum(
            r["status"] == "ok"
            for r in self.records
        )

        empty = sum(
            r["status"] == "empty"
            for r in self.records
        )

        failed = sum(
            r["status"] == "invalid_crop"
            for r in self.records
        )

        print("\n===== OCR DEBUG =====")
        print(f"Total bubbles : {total}")
        print(f"OCR success   : {ok}")
        print(f"OCR empty     : {empty}")
        print(f"Invalid crop  : {failed}")
        print("=====================\n")
