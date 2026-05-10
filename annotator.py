import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pygame


TRAIN_DIR = Path("chest_xray/train")
ANNOTATIONS_FILE = "annotations.csv"
CLASSES = ["NORMAL", "PNEUMONIA"]
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
TARGET_ANNOTATIONS = 20


class Annotator:
    def __init__(self):
        self.annotated = self._load_annotated()
        self.images = self._collect_images()
        self.total = len(self.images)
        self.current_idx = 0
        self.saved_count = 0

        pygame.init()
        pygame.display.set_caption("Annotator — 0/20 done")
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont("monospace", 20)
        self.clock = pygame.time.Clock()

        self.state = "IDLE"
        self.bbox_disp = None
        self.bbox_orig = None
        self.label_chosen = ""
        self.drag_start = None
        self.drag_end = None

        self.img_original = None
        self.img_surface = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.disp_w = 0
        self.disp_h = 0
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

    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _scale_to_fit(self, w, h):
        scale = min(WINDOW_WIDTH / w, WINDOW_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        ox = (WINDOW_WIDTH - new_w) // 2
        oy = (WINDOW_HEIGHT - new_h) // 2
        return scale, new_w, new_h, ox, oy

    def _render_image(self):
        h, w = self.img_original.shape[:2]
        surf = pygame.image.frombuffer(self.img_original.tobytes(), (w, h), "RGB")
        self.img_surface = pygame.transform.scale(surf, (self.disp_w, self.disp_h))

    def _clamp_display_coord(self, x, y):
        cx = max(0, min(x, self.disp_w - 1))
        cy = max(0, min(y, self.disp_h - 1))
        return cx, cy

    def _display_to_orig(self, dx, dy):
        return int(dx / self.scale), int(dy / self.scale)

    def _build_display(self):
        self.window.fill((30, 30, 30))
        self.window.blit(self.img_surface, (self.offset_x, self.offset_y))

        text = self.font.render(self.current_fname, True, (0, 255, 0))
        self.window.blit(text, (10, 10))

        progress = f"{self.saved_count} / {TARGET_ANNOTATIONS} done"
        text = self.font.render(progress, True, (0, 255, 0))
        self.window.blit(text, (WINDOW_WIDTH - text.get_width() - 10, 10))

        if self.state == "DRAGGING" and self.drag_start and self.drag_end:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            pygame.draw.rect(self.window, (255, 0, 0),
                             (x1 + self.offset_x, y1 + self.offset_y, x2 - x1, y2 - y1), 2)
        elif self.state == "BOX_DRAWN" and self.bbox_disp:
            x1, y1, x2, y2 = self.bbox_disp
            pygame.draw.rect(self.window, (255, 0, 0),
                             (x1 + self.offset_x, y1 + self.offset_y, x2 - x1, y2 - y1), 2)

        if self.state == "IDLE":
            text = self.font.render("Click and drag to draw bounding box", True, (255, 255, 0))
            self.window.blit(text, (10, WINDOW_HEIGHT - 30))
        elif self.state == "BOX_DRAWN":
            label_val = self.label_chosen if self.label_chosen else "?"
            text = f"[N] Normal  [P] Pneumonia  [S] Save  [R] Redo  [Q] Quit  Label: {label_val}"
            self.window.blit(self.font.render(text, True, (255, 255, 0)),
                             (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()

    def run(self):
        if self.total == 0:
            print("All images already annotated. Nothing to do.")
            pygame.quit()
            return

        print(f"Images available: {self.total}")
        print(f"Target: {TARGET_ANNOTATIONS} annotations")
        running = True

        while running and self.current_idx < self.total and self.saved_count < TARGET_ANNOTATIONS:
            img_path, self.current_cls, self.current_fname, self.rel_path = \
                self.images[self.current_idx]

            self.img_original = self._load_image(img_path)
            if self.img_original is None:
                self.current_idx += 1
                continue

            h, w = self.img_original.shape[:2]
            self.scale, self.disp_w, self.disp_h, self.offset_x, self.offset_y = \
                self._scale_to_fit(w, h)
            self._render_image()

            self.state = "IDLE"
            self.bbox_disp = None
            self.bbox_orig = None
            self.label_chosen = ""
            self.drag_start = None
            self.drag_end = None

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.state == "IDLE":
                            x = event.pos[0] - self.offset_x
                            y = event.pos[1] - self.offset_y
                            if 0 <= x < self.disp_w and 0 <= y < self.disp_h:
                                self.drag_start = (x, y)
                                self.drag_end = (x, y)
                                self.state = "DRAGGING"

                    elif event.type == pygame.MOUSEMOTION:
                        if self.state == "DRAGGING":
                            x = event.pos[0] - self.offset_x
                            y = event.pos[1] - self.offset_y
                            self.drag_end = self._clamp_display_coord(x, y)

                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        if self.state == "DRAGGING":
                            x = event.pos[0] - self.offset_x
                            y = event.pos[1] - self.offset_y
                            cx, cy = self._clamp_display_coord(x, y)
                            sx, sy = self.drag_start
                            x1, y1 = min(sx, cx), min(sy, cy)
                            x2, y2 = max(sx, cx), max(sy, cy)
                            if x2 - x1 < 5 or y2 - y1 < 5:
                                self.state = "IDLE"
                            else:
                                self.bbox_disp = (x1, y1, x2, y2)
                                ox1, oy1 = self._display_to_orig(x1, y1)
                                ox2, oy2 = self._display_to_orig(x2, y2)
                                self.bbox_orig = (ox1, oy1, ox2, oy2)
                                self.state = "BOX_DRAWN"
                            self.drag_start = None
                            self.drag_end = None

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                            break

                        if self.state == "BOX_DRAWN":
                            if event.key == pygame.K_n:
                                self.label_chosen = "NORMAL"
                            elif event.key == pygame.K_p:
                                self.label_chosen = "PNEUMONIA"
                            elif event.key == pygame.K_s:
                                if not self.label_chosen:
                                    continue
                                self._save_annotation(
                                    self.rel_path, self.label_chosen, *self.bbox_orig)
                                self.saved_count += 1
                                self.current_idx += 1
                                break
                            elif event.key == pygame.K_r:
                                self.state = "IDLE"
                                self.bbox_disp = None
                                self.bbox_orig = None
                                self.label_chosen = ""

                if not running:
                    break

                self._build_display()
                pygame.display.set_caption(
                    f"Annotator — {self.saved_count}/{TARGET_ANNOTATIONS} done")
                self.clock.tick(60)

        pygame.quit()
        if self.saved_count >= TARGET_ANNOTATIONS:
            print(f"\u2705 {TARGET_ANNOTATIONS} annotations complete!")
        else:
            print(f"Saved progress. {self.saved_count} annotations saved.")


if __name__ == "__main__":
    Annotator().run()
