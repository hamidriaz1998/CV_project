import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import pygame

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
GALLERY_WIDTH = 220
THUMB_W = 200
THUMB_H = 100
ITEM_H = THUMB_H + 28
GALLERY_PADDING = 8
MAIN_AREA_X = GALLERY_WIDTH
MAIN_AREA_W = WINDOW_WIDTH - GALLERY_WIDTH
MAIN_AREA_H = WINDOW_HEIGHT

TRAIN_DIR = Path("chest_xray/train/NORMAL")
ANNOTATIONS_FILE = Path("annotations.json")
CLASSES_FILE = Path("classes.json")

BG = (30, 30, 30)
GALLERY_BG = (40, 40, 45)
HIGHLIGHT = (0, 150, 255)
BOX_RED = (255, 0, 0)
BOX_PURPLE = (180, 0, 255)
TEXT_GREEN = (0, 255, 0)
HELP_YELLOW = (255, 255, 0)
OVERLAY_BG = (20, 20, 30, 200)
ANNOTATED_INDICATOR = (50, 200, 50)


class Annotator:
    def __init__(self, image_sources=None):
        self.image_sources = image_sources
        self._load_data()
        self.images = self._collect_images()
        self.total = len(self.images)
        self.current_idx = 0
        self.saved_count = 0

        pygame.init()
        pygame.display.set_caption("Annotator")
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_bold = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 13)
        self.font_input = pygame.font.SysFont("monospace", 24)
        self.clock = pygame.time.Clock()

        self.state = "IDLE"
        self.bbox_disp = None
        self.bbox_orig = None
        self.selected_class = ""
        self.drag_start = None
        self.drag_end = None

        self.img_original = None
        self.img_surface = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.disp_w = 0
        self.disp_h = 0
        self.current_rel_path = ""

        self.scroll_offset = 0
        self.thumbnail_cache: dict[str, pygame.Surface] = {}

        self.text_input = ""
        self.text_input_active = False

        self.message = ""
        self.message_timer = 0

    def _load_data(self):
        global_classes = ["NORMAL", "PNEUMONIA"]

        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE) as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
            for item in data:
                for key in ("x1", "y1", "x2", "y2"):
                    if key in item:
                        item[key] = int(item[key])
                item.setdefault("id", "")
                item.setdefault("created_at", "")
                for key in ("filename", "label", "x1", "y1", "x2", "y2"):
                    item.setdefault(key, "" if key in ("filename", "label") else 0)
            self.annotations = data
            self._next_ann_id = (
                max(int(a["id"]) for a in self.annotations if a["id"].isdigit()) + 1
                if self.annotations
                else 1
            )
        else:
            self.annotations = []
            self._next_ann_id = 1

        if CLASSES_FILE.exists():
            with open(CLASSES_FILE) as f:
                classes = json.load(f)
            self.classes = [c.upper().strip() for c in classes if c.strip()]
        else:
            self.classes = global_classes
            self._save_classes()

    def _save_annotations(self):
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def _save_classes(self):
        with open(CLASSES_FILE, "w") as f:
            json.dump(self.classes, f, indent=2)

    def _collect_images(self):
        images = []

        if self.image_sources:
            for src in self.image_sources:
                src = Path(src)
                if not src.is_dir():
                    continue
                cls_name = src.name
                for fname in sorted(os.listdir(src)):
                    if not fname.lower().endswith(
                        (".jpeg", ".jpg", ".png", ".bmp", ".webp")
                    ):
                        continue
                    rel_path = f"{cls_name}/{fname}"
                    images.append((src / fname, cls_name, fname, rel_path))
        else:
            if not TRAIN_DIR.exists():
                return images
            subdirs = [d for d in sorted(TRAIN_DIR.iterdir()) if d.is_dir()]
            if subdirs:
                for cls_dir in subdirs:
                    cls_name = cls_dir.name
                    for fname in sorted(os.listdir(cls_dir)):
                        if not fname.lower().endswith(
                            (".jpeg", ".jpg", ".png", ".bmp", ".webp")
                        ):
                            continue
                        rel_path = f"train/{cls_name}/{fname}"
                        images.append((cls_dir / fname, cls_name, fname, rel_path))
            else:
                cls_name = TRAIN_DIR.name
                for fname in sorted(os.listdir(TRAIN_DIR)):
                    if not fname.lower().endswith(
                        (".jpeg", ".jpg", ".png", ".bmp", ".webp")
                    ):
                        continue
                    rel_path = f"{cls_name}/{fname}"
                    images.append((TRAIN_DIR / fname, cls_name, fname, rel_path))

        return images

    def _get_image_annotations(self, rel_path, basename):
        result = []
        for a in self.annotations:
            fn = a.get("filename", "")
            if fn == rel_path or fn == basename:
                result.append(a)
        return result

    def _set_message(self, msg):
        self.message = msg
        self.message_timer = 120

    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _scale_to_fit(self, w, h):
        scale = min(MAIN_AREA_W / w, MAIN_AREA_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        ox = MAIN_AREA_X + (MAIN_AREA_W - new_w) // 2
        oy = (MAIN_AREA_H - new_h) // 2
        return scale, new_w, new_h, ox, oy

    def _render_image(self):
        h, w = self.img_original.shape[:2]
        surf = pygame.image.frombuffer(self.img_original.tobytes(), (w, h), "RGB")
        self.img_surface = pygame.transform.scale(surf, (self.disp_w, self.disp_h))

    def _clamp_display_coord(self, x, y):
        cx = max(0, min(x, self.disp_w - 1))
        cy = max(0, min(y, self.disp_h - 1))
        return int(cx), int(cy)

    def _display_to_orig(self, dx, dy):
        return int(dx / self.scale), int(dy / self.scale)

    def _get_thumbnail(self, img_path, fname):
        if fname in self.thumbnail_cache:
            return self.thumbnail_cache[fname]

        img = cv2.imread(str(img_path))
        if img is None:
            surf = pygame.Surface((THUMB_W, THUMB_H))
            surf.fill((60, 60, 60))
            self.thumbnail_cache[fname] = surf
            return surf

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(THUMB_W / w, THUMB_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        surf = pygame.image.frombuffer(img.tobytes(), (new_w, new_h), "RGB")
        self.thumbnail_cache[fname] = surf
        return surf

    def _get_annotated_basenames(self):
        result = set()
        for a in self.annotations:
            fn = a.get("filename", "")
            if "/" in fn:
                result.add(Path(fn).name)
            else:
                result.add(fn)
        return result

    def _render_gallery(self):
        pygame.draw.rect(self.window, GALLERY_BG, (0, 0, GALLERY_WIDTH, WINDOW_HEIGHT))

        max_scroll = max(0, self.total * ITEM_H + GALLERY_PADDING - WINDOW_HEIGHT)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

        annotated_basenames = self._get_annotated_basenames()
        clip_rect = pygame.Rect(0, 0, GALLERY_WIDTH, WINDOW_HEIGHT)
        self.window.set_clip(clip_rect)

        start_idx = max(0, self.scroll_offset // ITEM_H - 1)
        end_idx = min(self.total, (self.scroll_offset + WINDOW_HEIGHT) // ITEM_H + 2)

        for i in range(start_idx, end_idx):
            img_path, cls_name, fname, rel_path = self.images[i]
            y_pos = i * ITEM_H + GALLERY_PADDING - self.scroll_offset

            if y_pos + ITEM_H < 0 or y_pos > WINDOW_HEIGHT:
                continue

            thumb = self._get_thumbnail(img_path, fname)
            thumb_y = y_pos
            thumb_x = (GALLERY_WIDTH - THUMB_W) // 2
            self.window.blit(thumb, (thumb_x, thumb_y))

            is_annotated = (
                fname in annotated_basenames or rel_path in annotated_basenames
            )

            text = self.font_small.render(fname, True, TEXT_GREEN)
            text_y = thumb_y + THUMB_H + 2
            self.window.blit(text, (thumb_x + 2, text_y))

            if is_annotated:
                dot = self.font_small.render("●", True, ANNOTATED_INDICATOR)
                self.window.blit(dot, (GALLERY_WIDTH - 20, thumb_y + 2))

            if i == self.current_idx:
                pygame.draw.rect(
                    self.window,
                    HIGHLIGHT,
                    (thumb_x - 2, thumb_y - 2, THUMB_W + 4, ITEM_H + 4),
                    2,
                )

        self.window.set_clip(None)

    def _handle_gallery_click(self, pos):
        if pos[0] > GALLERY_WIDTH:
            return False

        clicked_idx = (pos[1] + self.scroll_offset - GALLERY_PADDING) // ITEM_H
        if 0 <= clicked_idx < self.total:
            self.current_idx = clicked_idx
            return True
        return False

    def _build_display(self):
        self.window.fill(BG)
        self._render_gallery()
        self.window.blit(self.img_surface, (self.offset_x, self.offset_y))

        existing_anns = self._get_image_annotations(
            self.current_rel_path,
            self.images[self.current_idx][2] if self.images else "",
        )
        for ann in existing_anns:
            x1 = int(ann["x1"] * self.scale) + self.offset_x
            y1 = int(ann["y1"] * self.scale) + self.offset_y
            x2 = int(ann["x2"] * self.scale) + self.offset_x
            y2 = int(ann["y2"] * self.scale) + self.offset_y
            pygame.draw.rect(self.window, BOX_PURPLE, (x1, y1, x2 - x1, y2 - y1), 2)
            label = self.font_small.render(ann["label"], True, BOX_PURPLE)
            self.window.blit(label, (x1 + 2, y1 - 14))

        if self.state == "DRAGGING" and self.drag_start and self.drag_end:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            x1, y1 = int(min(sx, ex)), int(min(sy, ey))
            x2, y2 = int(max(sx, ex)), int(max(sy, ey))
            pygame.draw.rect(
                self.window,
                BOX_RED,
                (x1 + self.offset_x, y1 + self.offset_y, x2 - x1, y2 - y1),
                2,
            )
        elif self.state == "BOX_DRAWN" and self.bbox_disp:
            x1, y1, x2, y2 = self.bbox_disp
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            pygame.draw.rect(
                self.window,
                BOX_RED,
                (x1 + self.offset_x, y1 + self.offset_y, x2 - x1, y2 - y1),
                2,
            )

        _, _, fname, rel_path = self.images[self.current_idx]
        text = self.font.render(fname, True, TEXT_GREEN)
        self.window.blit(text, (MAIN_AREA_X + 10, 10))

        progress = f"{self.saved_count} saved"
        text = self.font.render(progress, True, TEXT_GREEN)
        self.window.blit(text, (WINDOW_WIDTH - text.get_width() - 10, 10))

        total_annotated = len(self.annotations)
        text = self.font.render(f"Total: {total_annotated}", True, TEXT_GREEN)
        self.window.blit(text, (WINDOW_WIDTH - text.get_width() - 10, 30))

        classes_y = WINDOW_HEIGHT - 30 - len(self.classes) * 22
        for i, cls_name in enumerate(self.classes):
            prefix = f"[{i + 1}]"
            if cls_name == self.selected_class:
                prefix += " *"
            color = HELP_YELLOW if cls_name == self.selected_class else TEXT_GREEN
            line = f"{prefix} {cls_name}"
            text = self.font.render(line, True, color)
            self.window.blit(text, (MAIN_AREA_X + 10, classes_y + i * 22))

        help_y = WINDOW_HEIGHT - 30
        if self.state == "IDLE":
            text = self.font.render(
                "Click & drag to draw box | A/D: nav | Q: quit", True, HELP_YELLOW
            )
            self.window.blit(text, (MAIN_AREA_X + 10, help_y))
        elif self.state == "BOX_DRAWN":
            label_val = self.selected_class if self.selected_class else "?"
            parts = [
                f"Label: {label_val}",
                "[1-9]: Class",
                "[S]: Save",
                "[R]: Redo",
                "[C]: Add class",
                "[X]: Del class",
                "[E]: COCO export",
                "[Q]: Quit",
            ]
            for pi, part in enumerate(parts):
                text = self.font.render(part, True, HELP_YELLOW)
                self.window.blit(text, (MAIN_AREA_X + 10, help_y - pi * 20))

        if self.message and self.message_timer > 0:
            text = self.font_bold.render(self.message, True, HELP_YELLOW)
            x = (WINDOW_WIDTH - text.get_width()) // 2
            self.window.blit(text, (x, WINDOW_HEIGHT // 2))

        if self.text_input_active:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill(OVERLAY_BG)
            self.window.blit(overlay, (0, 0))

            prompt = self.font_bold.render("Enter class name:", True, HELP_YELLOW)
            input_surf = self.font_input.render(
                self.text_input + "|", True, (255, 255, 255)
            )
            px = (WINDOW_WIDTH - prompt.get_width()) // 2
            py = WINDOW_HEIGHT // 2 - 50
            self.window.blit(prompt, (px, py))
            self.window.blit(
                input_surf, ((WINDOW_WIDTH - input_surf.get_width()) // 2, py + 40)
            )
            hint = self.font.render("Enter: confirm  |  Esc: cancel", True, TEXT_GREEN)
            self.window.blit(hint, ((WINDOW_WIDTH - hint.get_width()) // 2, py + 80))

        pygame.display.flip()

    def _resolve_image_path(self, fname):
        if "/" in fname:
            p = TRAIN_DIR.parent / fname
            if p.exists():
                return p
        for cls_dir in TRAIN_DIR.iterdir():
            if cls_dir.is_dir():
                p = cls_dir / fname
                if p.exists():
                    return p
        return None

    def _get_image_size(self, path):
        if path is None:
            return (0, 0)
        try:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        except Exception:
            pass
        return (0, 0)

    def _export_coco(self):
        cat_names = sorted(set(a["label"] for a in self.annotations))
        cat_map = {name: i + 1 for i, name in enumerate(cat_names)}

        img_ids = {}
        images = []
        coco_annotations = []

        for a in self.annotations:
            fname = a["filename"]
            if fname not in img_ids:
                img_id = len(img_ids) + 1
                img_ids[fname] = img_id
                img_path = self._resolve_image_path(fname)
                w, h = self._get_image_size(img_path)
                images.append(
                    {
                        "id": img_id,
                        "file_name": fname,
                        "width": w,
                        "height": h,
                    }
                )

            img_id = img_ids[fname]
            bw = a["x2"] - a["x1"]
            bh = a["y2"] - a["y1"]
            ann_id = len(coco_annotations) + 1
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_map[a["label"]],
                    "bbox": [a["x1"], a["y1"], bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )

        categories = [
            {"id": cid, "name": name}
            for name, cid in sorted(cat_map.items(), key=lambda x: x[1])
        ]

        coco = {
            "images": images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        output_path = Path("annotations_coco.json")
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        self._set_message(f"Exported COCO -> {output_path}")

    def _export_json(self):
        data = []
        for a in self.annotations:
            data.append(
                {
                    "filename": a["filename"],
                    "label": a["label"],
                    "x1": a["x1"],
                    "y1": a["y1"],
                    "x2": a["x2"],
                    "y2": a["y2"],
                }
            )
        output_path = Path("annotations_export.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        self._set_message(f"Exported JSON -> {output_path}")

    def _export_csv(self):
        import csv

        output_path = Path("annotations_export.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "x1", "y1", "x2", "y2"])
            for a in self.annotations:
                writer.writerow(
                    [a["filename"], a["label"], a["x1"], a["y1"], a["x2"], a["y2"]]
                )
        self._set_message(f"Exported CSV -> {output_path}")

    def run(self):
        print(f"Images loaded: {self.total}")
        print(f"Classes: {', '.join(self.classes)}")
        print(f"Existing annotations: {len(self.annotations)}")
        running = True

        while running and 0 <= self.current_idx < self.total:
            img_path, self.current_cls, self.current_fname, self.current_rel_path = (
                self.images[self.current_idx]
            )

            self.img_original = self._load_image(img_path)
            if self.img_original is None:
                self.current_idx += 1
                continue

            h, w = self.img_original.shape[:2]
            self.scale, self.disp_w, self.disp_h, self.offset_x, self.offset_y = (
                self._scale_to_fit(w, h)
            )
            self._render_image()

            self.state = "IDLE"
            self.bbox_disp = None
            self.bbox_orig = None
            self.selected_class = ""
            self.drag_start = None
            self.drag_end = None

            while running:
                next_image = False
                if self.message_timer > 0:
                    self.message_timer -= 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

                    if self.text_input_active:
                        self._handle_text_input(event)
                        continue

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            if self._handle_gallery_click(event.pos):
                                next_image = True
                                break
                            if event.pos[0] >= MAIN_AREA_X and self.state == "IDLE":
                                x = event.pos[0] - self.offset_x
                                y = event.pos[1] - self.offset_y
                                if 0 <= x < self.disp_w and 0 <= y < self.disp_h:
                                    self.drag_start = (x, y)
                                    self.drag_end = (x, y)
                                    self.state = "DRAGGING"
                        elif event.button == 4:
                            self.scroll_offset = max(0, self.scroll_offset - ITEM_H)
                        elif event.button == 5:
                            self.scroll_offset += ITEM_H

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
                            x1, y1 = int(min(sx, cx)), int(min(sy, cy))
                            x2, y2 = int(max(sx, cx)), int(max(sy, cy))
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
                        elif event.key in (pygame.K_a, pygame.K_LEFT):
                            if self.current_idx > 0:
                                self.current_idx -= 1
                                next_image = True
                                break
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            if self.current_idx < self.total - 1:
                                self.current_idx += 1
                                next_image = True
                                break

                        if self.state == "BOX_DRAWN":
                            if event.key == pygame.K_s:
                                if not self.selected_class:
                                    self._set_message("Select a class first (1-9)")
                                    continue
                                self._save_annotation()
                                self.saved_count += 1
                                next_image = True
                                break
                            elif event.key == pygame.K_r:
                                self.state = "IDLE"
                                self.bbox_disp = None
                                self.bbox_orig = None
                                self.selected_class = ""

                        class_key = event.key - pygame.K_1
                        if 0 <= class_key < len(self.classes):
                            self.selected_class = self.classes[class_key]
                            if self.state == "IDLE":
                                self.state = "BOX_DRAWN"

                        if event.key == pygame.K_c:
                            self._start_text_input()
                        elif event.key == pygame.K_x:
                            self._delete_current_class()
                        elif event.key == pygame.K_e:
                            self._export_coco()
                        elif event.key == pygame.K_j:
                            self._export_json()
                        elif event.key == pygame.K_v:
                            self._export_csv()

                if not running:
                    break
                if next_image:
                    break

                self._build_display()
                pygame.display.set_caption(
                    f"Annotator — img {self.current_idx + 1}/{self.total} | "
                    f"{self.saved_count} saved"
                )
                self.clock.tick(60)

        pygame.quit()
        print(f"\nDone. {self.saved_count} annotations saved this session.")
        print(f"Total annotations: {len(self.annotations)}")

    def _save_annotation(self):
        entry = {
            "filename": self.current_rel_path,
            "label": self.selected_class,
            "x1": self.bbox_orig[0],
            "y1": self.bbox_orig[1],
            "x2": self.bbox_orig[2],
            "y2": self.bbox_orig[3],
            "id": str(self._next_ann_id),
            "created_at": datetime.now().isoformat(),
        }
        self._next_ann_id += 1
        self.annotations.append(entry)
        self._save_annotations()
        self._set_message(f"Saved: {self.selected_class}")

    def _start_text_input(self):
        self.text_input_active = True
        self.text_input = ""

    def _handle_text_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                name = self.text_input.upper().strip()
                if name:
                    if name in self.classes:
                        self._set_message(f"Class '{name}' already exists")
                    else:
                        self.classes.append(name)
                        self._save_classes()
                        cls_dir = TRAIN_DIR / name
                        cls_dir.mkdir(parents=True, exist_ok=True)
                        self._set_message(f"Added class: {name}")
                self.text_input_active = False
                self.text_input = ""
            elif event.key == pygame.K_ESCAPE:
                self.text_input_active = False
                self.text_input = ""
            elif event.key == pygame.K_BACKSPACE:
                self.text_input = self.text_input[:-1]
            else:
                if event.unicode and event.unicode.isprintable():
                    self.text_input += event.unicode

    def _delete_current_class(self):
        if not self.selected_class:
            self._set_message("Select a class first (1-9)")
            return
        if self.selected_class not in self.classes:
            return
        if len(self.classes) <= 1:
            self._set_message("Cannot delete the last class")
            return
        self.classes.remove(self.selected_class)
        self._save_classes()
        self._set_message(f"Removed class: {self.selected_class}")
        self.selected_class = ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate medical images")
    parser.add_argument(
        "image_sources", nargs="*",
        help="Directories to scan for images (each dir name = class label). "
             "Default: scans subdirectories of chest_xray/train/"
    )
    args = parser.parse_args()
    sources = [Path(p) for p in args.image_sources] if args.image_sources else None
    Annotator(image_sources=sources).run()
