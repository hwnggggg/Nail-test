# ─── 1. Imports ───────────────────────────────────────────────────────────────
import os, io, json, base64, requests
from pathlib import Path
import openai, pandas as pd
from PIL import Image
import pillow_heif; pillow_heif.register_heif_opener()
from pdf2image import convert_from_bytes
import gspread

from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ─── 2. Authenticate Google Drive / Sheets ────────────────────────────────────
SERVICE_ACCOUNT_FILE = "credentials.json"
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    with open(SERVICE_ACCOUNT_FILE, "w") as f:
        f.write(os.getenv("GOOGLE_CREDENTIALS_JSON"))

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
gc = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

# ─── 3. Load Sheet and Detect Image Column ────────────────────────────────────
SHEET_URL = "https://docs.google.com/spreadsheets/d/1YXc2VOYnQqyawQSMbn0MW8lu3iZjDdDzR7q-PHo5sOA"
ws = gc.open_by_url(SHEET_URL).get_worksheet(0)

def tidy_headers(df_):
    df_ = df_.rename(columns=lambda c: c.strip())
    df_.columns = df_.columns.str.replace(r"\s+", " ", regex=True)
    return df_

raw_df = tidy_headers(get_as_dataframe(ws, evaluate_formulas=True))
photo_col = next((c for c in raw_df.columns if "photo" in c.lower()), None)
if photo_col is None:
    raise RuntimeError(f"No column containing 'photo' found in: {raw_df.columns.tolist()}")

df = raw_df.dropna(subset=[photo_col]).reset_index(drop=True)

# ─── 4. Ensure Result Columns Exist ───────────────────────────────────────────
result_cols = [
    "Polish Application", "Cuticle Work", "Nail Shape",
    "Cleanliness", "Overall Score", "Recommendation"
]
for col in result_cols:
    if col not in df.columns:
        df[col] = None

# ─── 5. Image Helpers ─────────────────────────────────────────────────────────
def _download_drive_file(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def _to_jpeg(img_bytes):
    try:
        pages = convert_from_bytes(img_bytes, dpi=200, first_page=1, last_page=1)
        pil_img = pages[0].convert("RGB")
    except Exception:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()

def fetch_image_as_jpeg(path_or_url):
    try:
        file_id = None
        if "open?id=" in path_or_url:
            file_id = path_or_url.split("open?id=")[-1]
        elif "/file/d/" in path_or_url:
            file_id = path_or_url.split("/file/d/")[1].split("/")[0]

        if file_id:
            return _to_jpeg(_download_drive_file(file_id))

        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        return _to_jpeg(resp.content)

    except Exception as e:
        print(f"[fetch_image_as_jpeg] {e}")
        return None

b64 = lambda data: base64.b64encode(data).decode()

# ─── 6. GPT Prompt and Call ───────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = (
    "You are a professional nail technician recruiter. Given this nail job photo, "
    "for each of the four categories below, give me:\n"
    "  • a score from 1 to 10 (integers only)\n"
    "  • a very brief comment (2–4 words)\n\n"
    "Categories:\n"
    "  – Polish Application\n"
    "  – Cuticle Work\n"
    "  – Nail Shape\n"
    "  – Cleanliness\n\n"
    "Then compute the average of these four numeric scores as “Overall Score” (number only), "
    "and pick one recommendation:\n"
    "  • “Highly Recommend Hire” if ≥ 8.5\n"
    "  • “Recommend Hire” if ≥ 7\n"
    "  • “Further Training Required” if ≥ 5.5\n"
    "  • “Not Recommend Hire” if < 5.5\n\n"
    "Respond with ONLY a JSON object, no prose, no fences."
)

def get_nail_assessment(jpeg_b64):
    resp = openai.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}
            ]}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found: {raw!r}")
    return json.loads(raw[start:end+1])

# ─── 7. Run GPT per Row ───────────────────────────────────────────────────────
categories = ["Polish Application", "Cuticle Work", "Nail Shape", "Cleanliness"]
flat_results = []

key_map = {
    "PolishApplication": "Polish Application",
    "CuticleWork": "Cuticle Work",
    "NailShape": "Nail Shape",
    "Cleanliness": "Cleanliness",
    "OverallScore": "Overall Score",
    "Recommendation": "Recommendation"
}

for i, row in df.iterrows():
    jpeg = fetch_image_as_jpeg(row[photo_col])
    if not jpeg:
        print(f"Row {i}: image load failed.")
        flat_results.append({col: "None" for col in result_cols})
        continue

    try:
        res = get_nail_assessment(b64(jpeg))
        flattened = {}

        for key, col in key_map.items():
            val = res.get(key) or res.get(col) or "None"
            if isinstance(val, dict):
                flattened[col] = f'{val.get("score")}, {val.get("comment")}'
            else:
                flattened[col] = val

        flat_results.append(flattened)

    except Exception as e:
        print(f"Row {i}: GPT error – {e}")
        flat_results.append({col: "None" for col in result_cols})

# ─── 8. Write Back to Sheet ───────────────────────────────────────────────────
for i, result in enumerate(flat_results):
    for col, val in result.items():
        if col not in df.columns:
            df[col] = None
        df.at[i, col] = str(val)

set_with_dataframe(ws, df, include_index=False, resize=True)
print("✅ Completed updates!")

