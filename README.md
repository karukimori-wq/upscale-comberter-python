# StampCut — Python 鮮明化 API

## ファイル構成

```
stamp-cutter-api/
├── main.py           # FastAPI サーバー（鮮明化処理）
├── requirements.txt  # Python依存ライブラリ
├── Dockerfile        # Railway用
├── stamp-cutter.html # フロントエンド（API連携追加済み）
└── README.md
```

## Railwayへのデプロイ手順

### 1. GitHubにpush

```bash
git add .
git commit -m "Add Python sharpen API"
git push
```

### 2. Railwayでサービス作成

1. <https://railway.app> にアクセス
1. **New Project** → **Deploy from GitHub repo** を選択
1. このリポジトリを選択（`stamp-cutter-api/` フォルダを含む側）
1. 自動的に Dockerfile を検知してビルド開始

> ⚠️ Railwayはリポジトリ直下の `Dockerfile` を自動検知します。
> サブフォルダの場合は Railway の Settings → Source → Root Directory を `stamp-cutter-api` に設定してください。

### 3. URLを確認

デプロイ完了後、Railway ダッシュボードの **Settings → Domains** に表示される URL をコピー。

例: `https://stampcut-sharpen-production.up.railway.app`

### 4. フロントエンドに設定

`stamp-cutter.html` を Vercel でホストし、  
**Python 鮮明化 API** カードの API URL 欄に上記 URL を貼り付け。

-----

## API エンドポイント

### `GET /`

ヘルスチェック

### `POST /sharpen`

**Request:**

```json
{
  "image": "<Base64 PNG文字列>",
  "mode": "line",        // "line" | "strong"
  "strength": 1.0        // 0.0 〜 2.0
}
```

**Response:**

```json
{
  "image": "<処理済みBase64 PNG>",
  "width": 370,
  "height": 320
}
```

## 処理アルゴリズム

### `line` モード（イラスト・手書き線向け）

1. **アンシャープマスク** — エッジを鮮明に
1. **DoG（Difference of Gaussians）** — 輪郭線のみを暗く強調
1. **CLAHE** — 局所コントラスト強調

### `strong` モード（薄い線・スキャン向け）

1. **カーネルシャープ** — 強めのフィルタ
1. **ガンマ補正** — 線を濃く

-----

## VercelのCORS設定（本番）

`main.py` の `allow_origins` を本番 URL に限定することを推奨：

```python
allow_origins=["https://your-app.vercel.app"],
```