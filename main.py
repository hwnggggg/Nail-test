# ─── 1. Install and import tools ─────────────────────────────────────────────
import os, io, json, base64, requests
from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd
import openai
import pytz
import pillow_heif
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes

import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# Enable HEIF support
pillow_heif.register_heif_opener()

# ─── 2. Authenticate Google Drive / Sheets ───────────────────────────────────
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

# ─── 3. Load Sheet and Detect Image Column ───────────────────────────────────
SHEET_URL = "stop this flow"
sh = gc.open_by_url(SHEET_URL)
worksheet = sh.get_worksheet(0)

# Load DataFrame and drop empty photo rows
df = get_as_dataframe(worksheet).dropna(subset=["Photo"]).reset_index(drop=True)

# ─── 4. Fetch Image Bytes ────────────────────────────────────────────────────
def fetch_image_bytes(path_or_url):
    try:
        # Skip local drive paths (not supported outside Colab)
        if path_or_url.startswith("/content/drive/"):
            print("Local Colab drive path not supported in GitHub Actions.")
            return None

        # Handle Google Drive share URLs
        if "open?id=" in path_or_url:
            file_id = path_or_url.split("open?id=")[-1]
        elif "/file/d/" in path_or_url:
            file_id = path_or_url.split("/file/d/")[1].split("/")[0]
        else:
            raise ValueError("Unrecognized Photo entry")

        # Download image from Google Drive
        request = drive_service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)

        # Open image in RGB format (now supports HEIC too)
        img = Image.open(fh).convert("RGB")
        buf = BytesIO(); img.save(buf, format="JPEG")
        return buf.getvalue()

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Helper function to base64-encode image bytes
def encode_image_bytes(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

# ─── 5. GPT Assessment ───────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_nail_assessment(b64_img):
    prompt = (
        "You are a nail technician recruiter.\n"
        "If the nail job photo does NOT show exactly 1 dark-colored nail, 2 light-colored nails, "
        "and 2 French manicure nails (natural base with clean white tips), respond with this JSON:\n"
        "{\n"
        "  \"Polish Application\": \"0.0/10 – Wrong format\",\n"
        "  \"Cuticle Work\":       \"0.0/10 – Wrong format\",\n"
        "  \"Nail Shape\":         \"0.0/10 – Wrong format\",\n"
        "  \"Cleanliness\":        \"0.0/10 – Wrong format\",\n"
        "  \"Overall Score\":      0.0,\n"
        "  \"Recommendation\":     \"Wrong Format\"\n"
        "}\n"
        "Otherwise, assess the image and return a JSON object with:\n"
        "- Score out of 10 and short comment (2–4 words) for each:\n"
        "  • Polish Application\n"
        "  • Cuticle Work\n"
        "  • Nail Shape\n"
        "  • Cleanliness\n"
        "- Then compute average score as 'Overall Score'\n"
        "- And a 'Recommendation' from:\n"
        "  • 'Highly Recommend Hire' if ≥ 8.5\n"
        "  • 'Recommend Hire' if ≥ 7\n"
        "  • 'Further Training Required' if ≥ 5.5\n"
        "  • 'Not Recommend Hire' if < 5.5\n"
        "Respond ONLY with the JSON object. No extra text."
    )

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}
        ],
        max_tokens=200
    )

    raw = resp.choices[0].message.content.strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response: {raw!r}")
    json_str = raw[start:end+1]
    return json.loads(json_str)

# ─── 6. Process New Rows Only ────────────────────────────────────────────────
results = []
timestamp_col = "Timestamp Rating"

# Ensure Timestamp Rating column exists
if timestamp_col not in df.columns:
    df[timestamp_col] = ""

# Get current Boston (EDT) time
boston_now = lambda: datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d %H:%M:%S")

for idx, row in df.iterrows():
    # Skip already rated rows
    if pd.notnull(row.get(timestamp_col)) and str(row[timestamp_col]).strip() != "":
        print(f"Skipping row {idx} — already rated.")
        results.append(None)
        continue

    # Load and process image
    img_bytes = fetch_image_bytes(row["Photo"])
    if not img_bytes:
        print(f"Skipping row {idx}: could not load image.")
        results.append({
            "Polish Application": "Error",
            "Cuticle Work":       "Error",
            "Nail Shape":         "Error",
            "Cleanliness":        "Error",
            "Overall Score":      "Error",
            "Recommendation":     "Error",
            timestamp_col:        boston_now()
        })
        continue

    # Assess via GPT
    try:
        assessment = get_nail_assessment(encode_image_bytes(img_bytes))
        assessment[timestamp_col] = boston_now()
        results.append(assessment)
        print(f"✅ Row {idx} rated at {assessment[timestamp_col]}")
    except Exception as e:
        print(f"❌ Error on row {idx}: {e}")
        results.append({
            "Polish Application": "Error",
            "Cuticle Work":       "Error",
            "Nail Shape":         "Error",
            "Cleanliness":        "Error",
            "Overall Score":      "Error",
            "Recommendation":     "Error",
            timestamp_col:        boston_now()
        })

# ─── 7. Write Results to Google Sheet ────────────────────────────────────────
for i, res in enumerate(results):
    if res is None:
        continue
    for col, val in res.items():
        df.loc[i, col] = val

set_with_dataframe(worksheet, df)
print("✅ Completed all updates.")
