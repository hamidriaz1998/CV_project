import csv
import os
import random
from pathlib import Path

import cv2


TRAIN_DIR = Path("chest_xray/train")
ANNOTATIONS_FILE = "annotations.csv"
CLASSES = ["NORMAL", "PNEUMONIA"]
MAX_DISPLAY_HEIGHT = 900


class Annotator:
    def __init__(self):
        self.annotated = self._load_annotated()
        self.images = self._collect_images()
        self.total = len(self.images)
        self.current_idx = 0

        self.window_name = "Annotator"
        self.state = "IDLE"
        self.bbox_disp = None
        self.bbox_orig = None
        self.label_chosen = ""
        self.scale = 1.0
        self.drag_start = None
        self.drag_end = None
        self.base_display = None
        self.current_cls = ""
        self.current_fname = ""
        self.rel_path = ""

    def _load_annotated(self):
        if not os.path.exists(ANNOTATIONS_FILE):
            return set()
        with open(ANNOTATIONS_FILE) as f:
            reader = csv.DictReader(f)
            return {row["filename"] for row in reader}

    def _collect_images(self):
        images = []
        for cls in CLASSES:
            dirpath = TRAIN_DIR / cls
            if not dirpath.exists():
                continue
            for fname in sorted(os.listdir(dirpath)):
                if fname.lower().endswith((".jpeg", ".jpg")):
                    rel_path = f"train/{cls}/{fname}"
                    if rel_path not in self.annotated:
                        images.append((dirpath / fname, cls, fname, rel_path))
        random.shuffle(images)
        return images

    def _save_annotation(self, filename, label, x1, y1, x2, y2):
        write_header = not os.path.exists(ANNOTATIONS_FILE)
        with open(ANNOTATIONS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["filename", "label", "x1", "y1", "x2", "y2"])
            writer.writerow([filename, label, x1, y1, x2, y2])

    def mouse_callback(self, event, x, y, flags, param):
        if self.state == "IDLE":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.drag_end = (x, y)
                self.state = "DRAGGING"
        elif self.state == "DRAGGING":
            if event == cv2.EVENT_MOUSEMOVE:
                self.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = self.drag_start
                x2, y2 = x, y
                sx, sy = min(x1, x2), min(y1, y2)
                ex, ey = max(x1, x2), max(y1, y2)
                if ex - sx < 5 or ey - sy < 5:
                    self.state = "IDLE"
                else:
                    self.bbox_disp = (sx, sy, ex, ey)
                    if self.scale < 1.0:
                        self.bbox_orig = (
                            int(sx / self.scale), int(sy / self.scale),
                            int(ex / self.scale), int(ey / self.scale),
                        )
                    else:
                        self.bbox_orig = self.bbox_disp
                    self.state = "BOX_DRAWN"
                self.drag_start = None
                self.drag_end = None

    def _build_display(self):
        display = self.base_display.copy()
        h, w = display.shape[:2]

        cv2.putText(display, self.current_fname, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        label_text = f"Class: {self.current_cls}"
        (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(display, label_text, (w - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        active_box = None
        if self.state == "DRAGGING" and self.drag_start and self.drag_end:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            active_box = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
        elif self.state == "BOX_DRAWN" and self.bbox_disp:
            active_box = self.bbox_disp

        if active_box:
            cv2.rectangle(display, (active_box[0], active_box[1]),
                          (active_box[2], active_box[3]), (0, 0, 255), 2)

        if self.state == "IDLE":
            text = "Click and drag to draw bounding box"
        elif self.state == "DRAGGING":
            text = ""
        elif self.state == "BOX_DRAWN":
            label_val = self.label_chosen if self.label_chosen else "?"
            text = f"Label: [{label_val}]  N=NORMAL  P=PNEUMONIA  S=Save  R=Redo  Q=Quit"
        else:
            text = ""
        cv2.putText(display, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return display

    @staticmethod
    def _resize_for_display(img):
        h, w = img.shape[:2]
        if h <= MAX_DISPLAY_HEIGHT:
            return img.copy(), 1.0
        scale = MAX_DISPLAY_HEIGHT / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h)), scale

    def run(self):
        if self.total == 0:
            print("All images already annotated. Nothing to do.")
            return

        print(f"Images to annotate: {self.total}")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self.current_idx < self.total:
            img_path, self.current_cls, self.current_fname, self.rel_path = \
                self.images[self.current_idx]

            img = cv2.imread(str(img_path))
            if img is None:
                self.current_idx += 1
                continue

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.base_display, self.scale = self._resize_for_display(img)
            self.state = "IDLE"
            self.bbox_disp = None
            self.bbox_orig = None
            self.label_chosen = ""
            self.drag_start = None
            self.drag_end = None

            while True:
                display = self._build_display()
                title = f"Annotator — {self.current_idx + 1}/{self.total} done"
                cv2.setWindowTitle(self.window_name, title)
                cv2.imshow(self.window_name, display)

                key = cv2.waitKey(30) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    cv2.destroyAllWindows()
                    print(f"Saved progress. {self.current_idx}/{self.total} images annotated.")
                    return

                if self.state == "BOX_DRAWN":
                    if key == ord("n") or key == ord("N"):
                        self.label_chosen = "NORMAL"
                    elif key == ord("p") or key == ord("P"):
                        self.label_chosen = "PNEUMONIA"
                    elif key == ord("s") or key == ord("S"):
                        if not self.label_chosen:
                            continue
                        self._save_annotation(self.rel_path, self.label_chosen,
                                              *self.bbox_orig)
                        self.current_idx += 1
                        break
                    elif key == ord("r") or key == ord("R"):
                        self.state = "IDLE"
                        self.bbox_disp = None
                        self.bbox_orig = None
                        self.label_chosen = ""

        cv2.destroyAllWindows()
        print(f"All {self.total} images annotated!")


if __name__ == "__main__":
    Annotator().run()
