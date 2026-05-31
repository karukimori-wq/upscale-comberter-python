from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io
from mangum import Mangum


app = FastAPI(title="StampCut Sharpen API")

# CORS: VercelのURLを本番では限定する
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番: ["https://your-app.vercel.app"]
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class SharpenRequest(BaseModel):
    image: str          # Base64 PNG (data:image/png;base64,... or raw base64)
    mode: str = "line"  # "line" | "photo" | "strong"
    strength: float = 1.0  # 0.0〜2.0


class SharpenResponse(BaseModel):
    image: str          # Base64 PNG (raw, no prefix)
    width: int
    height: int


def decode_image(b64: str) -> np.ndarray:
    """Base64 → OpenCV BGR/BGRA numpy array"""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # アルファ保持
    if img is None:
        raise ValueError("画像のデコードに失敗しました")
    return img


def encode_image(img: np.ndarray) -> str:
    """OpenCV array → Base64 PNG"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("画像のエンコードに失敗しました")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def sharpen_line(img: np.ndarray, strength: float) -> np.ndarray:
    """
    手書き・イラスト向け鮮明化
    - アンシャープマスク（線のエッジを強調）
    - 適応的コントラスト強調（CLAHE）
    - DoG（Difference of Gaussians）によるエッジ追加強調
    """
    has_alpha = img.shape[2] == 4 if img.ndim == 3 else False

    if has_alpha:
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3]
    else:
        bgr = img.copy()
        alpha = None

    # ── Step 1: アンシャープマスク ──
    sigma = max(0.5, 1.0 * strength)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    amount = min(2.0, 1.5 * strength)
    sharpened = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)

    # ── Step 2: DoG でエッジ成分を暗く強調 ──
    if strength > 0.3:
        g1 = cv2.GaussianBlur(bgr, (0, 0), 0.8)
        g2 = cv2.GaussianBlur(bgr, (0, 0), 2.5)
        dog = g1.astype(np.int16) - g2.astype(np.int16)
        # 暗い方向（輪郭線）のみ加算
        mask = (dog < 0).astype(np.float32)
        ea = min(1.0, strength * 0.6)
        sharpened = np.clip(
            sharpened.astype(np.int16) + (dog * mask * ea * 1.5).astype(np.int16),
            0, 255
        ).astype(np.uint8)

    # ── Step 3: CLAHE（局所コントラスト強調）──
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = max(1.0, 2.0 * strength)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    sharpened = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── Step 4: アルファ復元 ──
    if has_alpha:
        result = cv2.merge([sharpened[:, :, 0],
                            sharpened[:, :, 1],
                            sharpened[:, :, 2],
                            alpha])
    else:
        result = sharpened

    return result


def sharpen_strong(img: np.ndarray, strength: float) -> np.ndarray:
    """強めのシャープ（線が薄い手書きスキャン向け）"""
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        bgr, alpha = img[:, :, :3].copy(), img[:, :, 3]
    else:
        bgr, alpha = img.copy(), None

    # カーネルによる強シャープ
    k = strength * 1.5
    kernel = np.array([
        [ 0,     -k,      0    ],
        [-k,  1+4*k,     -k    ],
        [ 0,     -k,      0    ]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(bgr, -1, kernel)

    # ガンマ補正で線を濃く
    gamma = 1.0 + strength * 0.8
    lut = np.array([min(255, int((v / 255.0) ** gamma * 255)) for v in range(256)], dtype=np.uint8)
    sharpened = cv2.LUT(sharpened, lut)

    if has_alpha:
        return cv2.merge([sharpened[:, :, 0], sharpened[:, :, 1], sharpened[:, :, 2], alpha])
    return sharpened


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
        if req.mode == "strong":
            result = sharpen_strong(img, strength)
        else:
            result = sharpen_line(img, strength)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"処理エラー: {e}")

    h, w = result.shape[:2]
    return SharpenResponse(
        image=encode_image(result),
        width=w,
        height=h,
    )
handler = Mangum(app)
