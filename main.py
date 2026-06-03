from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io

app = FastAPI(title="StampCut Sharpen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class SharpenRequest(BaseModel):
    image: str
    mode: str = "line"
    strength: float = 1.0
    # LINEスタンプサイズへのリサイズをAPI側で行うか
    resize_to_stamp: bool = False
    stamp_w: int = 370
    stamp_h: int = 320


class SharpenResponse(BaseModel):
    image: str
    width: int
    height: int


def decode_image(b64: str) -> np.ndarray:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像のデコードに失敗しました")
    return img


def encode_image(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("画像のエンコードに失敗しました")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def split_alpha(img: np.ndarray):
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        return img[:, :, :3].copy(), img[:, :, 3]
    return img.copy(), None


def merge_alpha(bgr: np.ndarray, alpha) -> np.ndarray:
    if alpha is None:
        return bgr
    return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])


def sharpen_illustration(img: np.ndarray, strength: float) -> np.ndarray:
    """
    カラーイラスト・LINEスタンプ向け総合鮮明化
    
    処理パイプライン：
    1. ノイズ除去（細部を守りながら）
    2. アンシャープマスク（輪郭くっきり）
    3. DoG エッジ強調（線を暗く太く）
    4. CLAHE（局所コントラスト）
    5. 彩度強調（色をビビッドに）
    6. 最終シャープ
    """
    bgr, alpha = split_alpha(img)
    s = strength  # 0.0〜2.0

    # ── Step 1: ノイズ除去（エッジ保護）──
    denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=20, sigmaSpace=5)

    # ── Step 2: アンシャープマスク（強め）──
    sigma1 = max(0.5, 0.8 * s)
    blur1 = cv2.GaussianBlur(denoised, (0, 0), sigma1)
    amount1 = min(3.0, 2.0 * s)
    sharpened = cv2.addWeighted(denoised, 1.0 + amount1, blur1, -amount1, 0)

    # ── Step 3: DoG エッジ強調（輪郭線を暗く）──
    if s > 0.2:
        g1 = cv2.GaussianBlur(denoised, (0, 0), 0.6)
        g2 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        dog = g1.astype(np.int16) - g2.astype(np.int16)
        edge_mask = (dog < 0).astype(np.float32)
        ea = min(1.2, s * 0.8)
        sharpened = np.clip(
            sharpened.astype(np.int16) + (dog * edge_mask * ea * 2.0).astype(np.int16),
            0, 255
        ).astype(np.uint8)

    # ── Step 4: CLAHE（局所コントラスト強調）──
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a_ch, b_ch = cv2.split(lab)
    clip = max(1.5, 3.0 * s)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    sharpened = cv2.cvtColor(cv2.merge([l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── Step 5: 彩度強調（色をビビッドに）──
    if s > 0.3:
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_boost = min(1.5, 1.0 + s * 0.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
        sharpened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Step 6: 最終アンシャープ（仕上げ）──
    if s > 0.5:
        blur2 = cv2.GaussianBlur(sharpened, (0, 0), 0.5)
        amount2 = min(1.5, s * 0.8)
        sharpened = cv2.addWeighted(sharpened, 1.0 + amount2, blur2, -amount2, 0)

    return merge_alpha(sharpened, alpha)


def sharpen_line_art(img: np.ndarray, strength: float) -> np.ndarray:
    """
    線画・手書き向け：線を濃く細くはっきり
    """
    bgr, alpha = split_alpha(img)
    s = strength

    # ── カーネルシャープ ──
    k = min(3.0, s * 2.0)
    kernel = np.array([
        [-k/4, -k/4, -k/4],
        [-k/4, 1+2*k, -k/4],
        [-k/4, -k/4, -k/4]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(bgr, -1, kernel)

    # ── CLAHE ──
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(2.0, 4.0 * s), tileGridSize=(8, 8))
    l = clahe.apply(l)
    sharpened = cv2.cvtColor(cv2.merge([l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── ガンマ補正（線を濃く）──
    gamma = 1.0 + s * 1.0
    lut = np.array([min(255, int((v / 255.0) ** gamma * 255)) for v in range(256)], dtype=np.uint8)
    sharpened = cv2.LUT(sharpened, lut)

    return merge_alpha(sharpened, alpha)


def resize_to_stamp(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """
    LINEスタンプサイズへ高品質リサイズ
    段階的縮小でジャギーを防ぐ
    """
    h, w = img.shape[:2]
    aspect = w / h

    # フィット計算
    if w / max_w > h / max_h:
        tw, th = max_w, max(1, round(max_w / aspect))
    else:
        th, tw = max_h, max(1, round(max_h * aspect))

    # 段階的縮小
    cur = img.copy()
    cw, ch = w, h
    while cw > tw * 2 or ch > th * 2:
        cw = max(tw, cw // 2)
        ch = max(th, ch // 2)
        cur = cv2.resize(cur, (cw, ch), interpolation=cv2.INTER_AREA)

    # 最終リサイズ
    resized = cv2.resize(cur, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    # キャンバスに配置
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
    else:
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)

    ox = (max_w - tw) // 2
    oy = (max_h - th) // 2
    canvas[oy:oy+th, ox:ox+tw] = resized
    return canvas


@app.get("/")
def health():
    return {"status": "ok", "service": "StampCut Sharpen API"}


@app.post("/sharpen", response_model=SharpenResponse)
def sharpen(req: SharpenRequest):
    try:
        img = decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像デコードエラー: {e}")

    strength = max(0.0, min(2.0, req.strength))

    try:
        # ── 鮮明化 ──
        if req.mode == "line":
            result = sharpen_line_art(img, strength)
        else:
            result = sharpen_illustration(img, strength)

        # ── リサイズ（オプション）──
        if req.resize_to_stamp:
            result = resize_to_stamp(result, req.stamp_w, req.stamp_h)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理エラー: {e}")

    h, w = result.shape[:2]
    return SharpenResponse(image=encode_image(result), width=w, height=h)
