// ===== State =====
const state = {
  images: [],
  annotations: [],
  classes: [],
  selectedImage: null,
  selectedClass: null,
  isDrawing: false,
  dragStart: null,
  dragEnd: null,
  currentBox: null,
  savedBoxes: [],
  displayScale: 1,
  imgOriginalWidth: 0,
  imgOriginalHeight: 0,
};

// ===== DOM refs =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const canvas = $('#annotateCanvas');
const ctx = canvas.getContext('2d');
const canvasWrapper = $('#canvasWrapper');
const canvasPlaceholder = $('#canvasPlaceholder');
const canvasStatus = $('#canvasStatus');
const gallery = $('#gallery');
const imageCount = $('#imageCount');
const classList = $('#classList');
const annotationList = $('#annotationList');
const annCount = $('#annCount');
const saveBtn = $('#saveBtn');
const clearBtn = $('#clearBtn');
const addClassBtn = $('#addClassBtn');
const newClassName = $('#newClassName');
const exportBtn = $('#exportBtn');
const exportFormat = $('#exportFormat');
const uploadArea = $('#uploadArea');
const fileInput = $('#fileInput');
const statusText = $('#statusText');
const drawInfo = $('#drawInfo');
const themeToggle = $('#themeToggle');

// ===== API =====
const API = {
  async getImages() {
    const res = await fetch('/images');
    if (!res.ok) throw new Error('Failed to load images');
    return res.json();
  },

  async uploadFiles(files) {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    const res = await fetch('/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
  },

  async getAnnotations() {
    const res = await fetch('/annotations');
    if (!res.ok) throw new Error('Failed to load annotations');
    return res.json();
  },

  async saveAnnotation(data) {
    const res = await fetch('/annotate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to save annotation');
    }
    return res.json();
  },

  async deleteAnnotation(id) {
    const res = await fetch(`/annotate/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete annotation');
  },

  async getClasses() {
    const res = await fetch('/classes');
    if (!res.ok) throw new Error('Failed to load classes');
    return res.json();
  },

  async addClass(name) {
    const res = await fetch(`/classes?name=${encodeURIComponent(name)}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to add class');
    }
    return res.json();
  },

  async removeClass(name) {
    const res = await fetch(`/classes/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to remove class');
  },

  exportURL(format) {
    return `/export?format=${format}`;
  },
};

// ===== Toast =====
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

// ===== Canvas Drawing =====
function fitImage(img, maxW, maxH) {
  const scale = Math.min(maxW / img.width, maxH / img.height, 1);
  return {
    width: Math.round(img.width * scale),
    height: Math.round(img.height * scale),
    scale,
  };
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawImageOnCanvas(img) {
  const maxW = canvasWrapper.clientWidth - 20;
  const maxH = canvasWrapper.clientHeight - 20;
  const fit = fitImage(img, maxW, maxH);

  canvas.width = fit.width;
  canvas.height = fit.height;
  state.displayScale = fit.scale;
  state.imgOriginalWidth = img.width;
  state.imgOriginalHeight = img.height;

  ctx.drawImage(img, 0, 0, fit.width, fit.height);
}

function drawBox(x1, y1, x2, y2, color = '#ff4444', label = '', dashed = false) {
  ctx.save();
  if (dashed) ctx.setLineDash([4, 4]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  if (label) {
    ctx.fillStyle = color;
    const txt = ctx.measureText(label);
    const pad = 2;
    const tw = txt.width + pad * 2;
    ctx.fillRect(x1, y1 - 20, tw, 20);
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.fillText(label, x1 + pad, y1 - 6);
  }
  ctx.restore();
}

function drawScene() {
  clearCanvas();
  if (!state.selectedImage) return;

  const img = state.selectedImage.img;
  drawImageOnCanvas(img);

  for (const box of state.savedBoxes) {
    const sx = box.x1 * state.displayScale;
    const sy = box.y1 * state.displayScale;
    const ex = box.x2 * state.displayScale;
    const ey = box.y2 * state.displayScale;
    drawBox(sx, sy, ex, ey, '#4ade80', box.label);
  }

  if (state.currentBox) {
    const b = state.currentBox;
    drawBox(b.x1, b.y1, b.x2, b.y2, '#ff4444', state.selectedClass || '', true);
  }

  if (state.isDrawing && state.dragStart && state.dragEnd) {
    const sx = state.dragStart.x;
    const sy = state.dragStart.y;
    const ex = state.dragEnd.x;
    const ey = state.dragEnd.y;
    const x1 = Math.min(sx, ex);
    const y1 = Math.min(sy, ey);
    const x2 = Math.max(sx, ex);
    const y2 = Math.max(sy, ey);
    drawBox(x1, y1, x2, y2, '#ffaa00', null, true);
  }
}

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

// ===== Canvas Event Handlers =====
canvas.addEventListener('mousedown', (e) => {
  if (!state.selectedImage || e.button !== 0) return;
  const coords = getCanvasCoords(e);
  if (coords.x < 0 || coords.y < 0 || coords.x > canvas.width || coords.y > canvas.height) return;
  state.isDrawing = true;
  state.dragStart = coords;
  state.dragEnd = coords;
  state.currentBox = null;
});

canvas.addEventListener('mousemove', (e) => {
  if (!state.isDrawing) return;
  state.dragEnd = getCanvasCoords(e);
  drawScene();
});

canvas.addEventListener('mouseup', (e) => {
  if (!state.isDrawing) return;
  state.isDrawing = false;
  const end = getCanvasCoords(e);
  state.dragEnd = end;

  if (!state.dragStart) return;
  const sx = state.dragStart.x;
  const sy = state.dragStart.y;
  const ex = state.dragEnd.x;
  const ey = state.dragEnd.y;
  const x1 = Math.min(sx, ex);
  const y1 = Math.min(sy, ey);
  const x2 = Math.max(sx, ex);
  const y2 = Math.max(sy, ey);

  if (x2 - x1 < 5 || y2 - y1 < 5) {
    state.dragStart = null;
    state.dragEnd = null;
    drawScene();
    return;
  }

  state.currentBox = { x1, y1, x2, y2 };
  state.dragStart = null;
  state.dragEnd = null;
  drawScene();
  updateSaveButton();
});

// ===== Update UI =====
function updateSaveButton() {
  saveBtn.disabled = !(state.currentBox && state.selectedClass);
}

function displayToOrig(dispX, dispY) {
  return {
    x: Math.round(dispX / state.displayScale),
    y: Math.round(dispY / state.displayScale),
  };
}

// ===== Image Loading =====
async function loadImageToCanvas(filename) {
  const existing = state.images.find((i) => i.filename === filename);
  if (!existing) return;

  state.selectedImage = existing;
  canvasPlaceholder.classList.add('hidden');
  canvas.style.display = 'block';
  canvasStatus.textContent = `Annotating: ${filename}`;

  const img = new Image();
  img.onload = () => {
    existing.img = img;
    state.savedBoxes = state.annotations.filter((a) => a.filename === filename);
    state.currentBox = null;
    state.isDrawing = false;
    drawScene();
    updateSaveButton();
    renderAnnotationList();
    renderGallery();
  };
  img.onerror = () => {
    toast('Failed to load image', 'error');
  };
  img.src = `/images/${encodeURIComponent(filename)}`;
}

// ===== Gallery =====
function renderGallery() {
  if (state.images.length === 0) {
    gallery.innerHTML = '<div class="gallery-empty">No images uploaded yet</div>';
    imageCount.textContent = '0';
    return;
  }

  imageCount.textContent = state.images.length;
  const annFilenames = new Set(state.annotations.map((a) => a.filename));

  gallery.innerHTML = state.images
    .map(
      (img) => `
      <div class="gallery-item ${state.selectedImage?.filename === img.filename ? 'active' : ''} ${annFilenames.has(img.filename) ? 'has-ann' : ''}" data-filename="${img.filename}">
        <img src="/images/${encodeURIComponent(img.filename)}" alt="${img.filename}" loading="lazy">
        <div class="check-overlay">&#10003;</div>
      </div>
    `
    )
    .join('');

  gallery.querySelectorAll('.gallery-item').forEach((el) => {
    el.addEventListener('click', () => {
      const fn = el.dataset.filename;
      if (fn !== state.selectedImage?.filename) {
        loadImageToCanvas(fn);
      }
    });
  });
}

// ===== Classes =====
function renderClassList() {
  classList.innerHTML = state.classes
    .map(
      (cls) => `
      <span class="class-chip ${state.selectedClass === cls ? 'selected' : ''}" data-class="${cls}">
        ${cls}
        <span class="remove-class" data-remove="${cls}">&times;</span>
      </span>
    `
    )
    .join('');

  classList.querySelectorAll('.class-chip').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.classList.contains('remove-class')) return;
      state.selectedClass = el.dataset.class;
      renderClassList();
      updateSaveButton();
      drawInfo.textContent = `Class: ${state.selectedClass}`;
    });
  });

  classList.querySelectorAll('.remove-class').forEach((el) => {
    el.addEventListener('click', async (e) => {
      e.stopPropagation();
      const cls = el.dataset.remove;
      try {
        await API.removeClass(cls);
        state.classes = state.classes.filter((c) => c !== cls);
        if (state.selectedClass === cls) {
          state.selectedClass = null;
          updateSaveButton();
          drawInfo.textContent = 'Select a class, then draw a box on the image';
        }
        renderClassList();
      } catch (err) {
        toast(err.message, 'error');
      }
    });
  });
}

// ===== Annotations List =====
function renderAnnotationList() {
  const current = state.selectedImage
    ? state.annotations.filter((a) => a.filename === state.selectedImage.filename)
    : [];

  annCount.textContent = current.length;

  if (current.length === 0) {
    annotationList.innerHTML = '<div class="ann-empty">No annotations for this image</div>';
    return;
  }

  const colorMap = {};
  state.classes.forEach((c, i) => {
    const hue = (i * 137.5) % 360;
    colorMap[c] = `hsl(${hue}, 55%, 50%)`;
  });

  annotationList.innerHTML = current
    .map(
      (ann) => `
      <div class="ann-item">
        <span class="ann-label" style="background:${colorMap[ann.label] || '#888'};color:#fff">${ann.label}</span>
        <span class="ann-coords">(${ann.x1}, ${ann.y1}) - (${ann.x2}, ${ann.y2})</span>
        <button class="btn-danger" data-id="${ann.id}" title="Delete annotation">&times;</button>
      </div>
    `
    )
    .join('');

  annotationList.querySelectorAll('.btn-danger').forEach((el) => {
    el.addEventListener('click', async () => {
      const id = el.dataset.id;
      try {
        await API.deleteAnnotation(id);
        state.annotations = state.annotations.filter((a) => a.id !== id);
        if (state.selectedImage) {
          state.savedBoxes = state.annotations.filter(
            (a) => a.filename === state.selectedImage.filename
          );
          drawScene();
        }
        renderAnnotationList();
        renderGallery();
        toast('Annotation deleted', 'success');
      } catch (err) {
        toast(err.message, 'error');
      }
    });
  });
}

// ===== Save Annotation =====
saveBtn.addEventListener('click', async () => {
  if (!state.currentBox || !state.selectedClass || !state.selectedImage) return;

  const orig1 = displayToOrig(state.currentBox.x1, state.currentBox.y1);
  const orig2 = displayToOrig(state.currentBox.x2, state.currentBox.y2);

  try {
    const data = {
      filename: state.selectedImage.filename,
      label: state.selectedClass,
      x1: orig1.x,
      y1: orig1.y,
      x2: orig2.x,
      y2: orig2.y,
    };
    const saved = await API.saveAnnotation(data);
    state.annotations.push(saved);
    state.savedBoxes = state.annotations.filter(
      (a) => a.filename === state.selectedImage.filename
    );
    state.currentBox = null;
    drawScene();
    updateSaveButton();
    renderAnnotationList();
    renderGallery();
    toast('Annotation saved!', 'success');
  } catch (err) {
    toast(err.message, 'error');
  }
});

// ===== Clear =====
clearBtn.addEventListener('click', () => {
  state.currentBox = null;
  state.isDrawing = false;
  state.dragStart = null;
  state.dragEnd = null;
  drawScene();
  updateSaveButton();
});

// ===== Export =====
exportBtn.addEventListener('click', () => {
  const fmt = exportFormat.value;
  if (state.annotations.length === 0) {
    toast('No annotations to export', 'error');
    return;
  }
  window.open(API.exportURL(fmt), '_blank');
});

// ===== Add Class =====
addClassBtn.addEventListener('click', async () => {
  const name = newClassName.value.trim().toUpperCase();
  if (!name) {
    toast('Enter a class name', 'error');
    return;
  }
  if (state.classes.includes(name)) {
    toast('Class already exists', 'error');
    return;
  }
  try {
    state.classes = await API.addClass(name);
    newClassName.value = '';
    renderClassList();
    toast(`Class "${name}" added`, 'success');
  } catch (err) {
    toast(err.message, 'error');
  }
});

newClassName.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') addClassBtn.click();
});

// ===== Upload =====
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', async (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter((f) =>
    /\.(jpg|jpeg|png|bmp|webp)$/i.test(f.name)
  );
  if (files.length === 0) {
    toast('No valid image files found', 'error');
    return;
  }
  await doUpload(files);
});

fileInput.addEventListener('change', async () => {
  const files = Array.from(fileInput.files);
  if (files.length === 0) return;
  await doUpload(files);
  fileInput.value = '';
});

async function doUpload(files) {
  statusText.textContent = `Uploading ${files.length} file(s)...`;
  try {
    const result = await API.uploadFiles(files);
    toast(`${result.uploaded.length} image(s) uploaded`, 'success');
    await refreshImages();
    if (result.uploaded.length > 0 && !state.selectedImage) {
      loadImageToCanvas(result.uploaded[0]);
    }
  } catch (err) {
    toast(err.message, 'error');
  }
  statusText.textContent = 'Ready';
}

// ===== Refresh =====
async function refreshImages() {
  try {
    state.images = await API.getImages();
    renderGallery();
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function refreshAnnotations() {
  try {
    state.annotations = await API.getAnnotations();
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function refreshClasses() {
  try {
    state.classes = await API.getClasses();
    renderClassList();
  } catch (err) {
    toast(err.message, 'error');
  }
}

// ===== Theme =====
function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  themeToggle.textContent = saved === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19';
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  themeToggle.textContent = next === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19';
});

// ===== Init =====
async function init() {
  initTheme();
  try {
    await Promise.all([refreshImages(), refreshAnnotations(), refreshClasses()]);
    if (state.images.length > 0) {
      await loadImageToCanvas(state.images[0].filename);
    }
  } catch (err) {
    toast(err.message, 'error');
  }

  window.addEventListener('resize', () => {
    if (state.selectedImage && state.selectedImage.img) {
      drawScene();
    }
  });
}

init();
