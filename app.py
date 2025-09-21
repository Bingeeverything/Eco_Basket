# Receipt CO2 Calculator
import os
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# App configuration
st.set_page_config(page_title="Receipt CO2 Calculator", page_icon="🌱", layout="centered")
st.title("🌱 Receipt Carbon Calculator")
load_dotenv()

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip() or "contact@example.com"

with st.expander("How this works", expanded=False):
    st.markdown("""
**Process:** Upload receipt photo or paste text > match items to database > calculate CO2 emissions

**Features:**
- Built-in emissions database with 80+ grocery items
- OCR text extraction from images and PDFs
- Fuzzy string matching to handle variations in item names
- Units: kg, L, or "each" - quantities can be edited before calculation
""")

# OCR setup
TESS_AVAILABLE = False
def ocr_image(_img: Image.Image) -> str:
    raise RuntimeError("Tesseract not available")

try:
    import pytesseract
    from PIL import ImageOps, ImageFilter

    # Allowing overide of the path
    candidates = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract", 
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract"
    ]
    
    tesseract_found = False
    for p in candidates:
        if p and Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            tesseract_found = True
            break
    
    # If not found in candidates, try to use system PATH
    if not tesseract_found:
        try:
            # Test if tesseract is available in PATH
            pytesseract.image_to_string(Image.new('RGB', (1, 1)), config="--version")
            tesseract_found = True
        except Exception:
            pass

    def preprocess_for_ocr(img: Image.Image) -> Image.Image:
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.MedianFilter(size=3))
        bw = g.point(lambda x: 255 if x > 160 else 0, mode="1")
        return bw

    def ocr_image(img: Image.Image) -> str:
        cfg = "--psm 6 --oem 1 -l eng"
        return pytesseract.image_to_string(preprocess_for_ocr(img), config=cfg)

    TESS_AVAILABLE = tesseract_found
except Exception:
    TESS_AVAILABLE = False

# ----------------------------
# Load emissions DB 
# ----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "emissions_food.csv"


FALLBACK_DB = pd.DataFrame([
    {"name":"beef mince","basis":"kg","ef":27.0,"syn":"beef|minced beef|mince|ground beef","category":"protein","alt":"chicken breast","alt_basis":"kg","alt_ef":6.9,"alt_note":"Swap beef→chicken (~-75%)","source":"OWID/Poore&Nemecek 2018"},
    {"name":"beef steak","basis":"kg","ef":27.0,"syn":"beef steak|sirloin|scotch fillet","category":"protein","alt":"chicken breast","alt_basis":"kg","alt_ef":6.9,"alt_note":"Swap beef→chicken (~-75%)","source":"OWID/Poore&Nemecek 2018"},
    {"name":"lamb","basis":"kg","ef":39.0,"syn":"lamb|mutton","category":"protein","alt":"chicken breast","alt_basis":"kg","alt_ef":6.9,"alt_note":"Swap lamb→chicken (~-82%)","source":"OWID/Poore&Nemecek 2018"},
    {"name":"pork","basis":"kg","ef":7.0,"syn":"pork|pork loin|pork mince","category":"protein","alt":"chicken breast","alt_basis":"kg","alt_ef":6.9,"alt_note":"Small reduction vs pork","source":"OWID/Poore&Nemecek 2018"},
    {"name":"chicken breast","basis":"kg","ef":6.9,"syn":"chicken|chicken breast|chicken fillet","category":"protein","alt":"tofu","alt_basis":"kg","alt_ef":3.0,"alt_note":"Try plant protein","source":"OWID/Poore&Nemecek 2018"},
    {"name":"fish fillet","basis":"kg","ef":6.0,"syn":"fish|salmon|barramundi|white fish","category":"protein","alt":"tofu","alt_basis":"kg","alt_ef":3.0,"alt_note":"Plant option halves CO₂","source":"OWID/Poore&Nemecek 2018"},
    {"name":"tofu","basis":"kg","ef":3.0,"syn":"tofu|bean curd","category":"protein","alt":"tempeh","alt_basis":"kg","alt_ef":3.2,"alt_note":"Similar footprint","source":"OWID/Poore&Nemecek 2018"},
    {"name":"tempeh","basis":"kg","ef":3.2,"syn":"tempeh","category":"protein","alt":"tofu","alt_basis":"kg","alt_ef":3.0,"alt_note":"Similar footprint","source":"OWID/Poore&Nemecek 2018"},
    {"name":"eggs (dozen)","basis":"ea","ef":0.7,"syn":"eggs|dozen eggs|12pk eggs","category":"protein","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"milk (dairy)","basis":"L","ef":1.3,"syn":"milk|full cream milk|dairy milk|cow milk","category":"dairy","alt":"oat milk","alt_basis":"L","alt_ef":0.4,"alt_note":"Oat milk ≈-70% vs dairy","source":"OWID/Poore&Nemecek 2018"},
    {"name":"oat milk","basis":"L","ef":0.4,"syn":"oat milk|oat beverage","category":"dairy","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"soy milk","basis":"L","ef":0.9,"syn":"soy milk|soya milk","category":"dairy","alt":"oat milk","alt_basis":"L","alt_ef":0.4,"alt_note":"Oat milk lower vs soy","source":"OWID/Poore&Nemecek 2018"},
    {"name":"almond milk","basis":"L","ef":0.7,"syn":"almond milk","category":"dairy","alt":"oat milk","alt_basis":"L","alt_ef":0.4,"alt_note":"Oat milk lower water use","source":"OWID/Poore&Nemecek 2018"},
    {"name":"cheese","basis":"kg","ef":13.6,"syn":"cheese|cheddar|tasty cheese|mozzarella","category":"dairy","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"yoghurt","basis":"kg","ef":2.2,"syn":"yoghurt|yogurt","category":"dairy","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"rice (white)","basis":"kg","ef":2.7,"syn":"rice|white rice|basmati|jasmine","category":"grains","alt":"lentils (dry)","alt_basis":"kg","alt_ef":0.9,"alt_note":"Swap to pulses (~-65%)","source":"OWID/Poore&Nemecek 2018"},
    {"name":"pasta","basis":"kg","ef":1.8,"syn":"pasta|spaghetti|penne|fusilli","category":"grains","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"bread loaf","basis":"ea","ef":0.5,"syn":"bread|loaf|sandwich bread|toast","category":"bakery","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"lentils (dry)","basis":"kg","ef":0.9,"syn":"lentils|dal|masoor","category":"pantry","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"chickpeas (dry)","basis":"kg","ef":0.9,"syn":"chickpeas|chana|garbanzo","category":"pantry","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"beans (canned)","basis":"kg","ef":1.5,"syn":"beans|kidney beans|black beans|baked beans","category":"pantry","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"apples","basis":"kg","ef":0.5,"syn":"apple|apples","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"bananas","basis":"kg","ef":0.8,"syn":"banana|bananas","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"tomatoes","basis":"kg","ef":1.1,"syn":"tomato|tomatoes","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"potatoes","basis":"kg","ef":0.4,"syn":"potato|potatoes","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"onions","basis":"kg","ef":0.5,"syn":"onion|onions","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"carrots","basis":"kg","ef":0.3,"syn":"carrot|carrots","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"lettuce","basis":"kg","ef":0.9,"syn":"lettuce|iceberg|cos|romaine","category":"produce","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"plastic bag","basis":"ea","ef":0.01,"syn":"plastic bag|checkout bag|carry bag","category":"household","alt":"reusable bag","alt_basis":"ea","alt_ef":0.05,"alt_note":"Needs 5–10 reuses to beat plastic","source":"est."},
    {"name":"reusable bag","basis":"ea","ef":0.05,"syn":"reusable bag|tote bag|fabric bag","category":"household","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"est."},
    {"name":"sugar","basis":"kg","ef":0.7,"syn":"sugar|white sugar|raw sugar","category":"pantry","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
    {"name":"chocolate","basis":"kg","ef":19.0,"syn":"chocolate|dark chocolate|milk chocolate","category":"snacks","alt":"","alt_basis":"","alt_ef":"","alt_note":"","source":"OWID/Poore&Nemecek 2018"},
])

def load_emissions_db(csv_path, fallback_df):
    """Load emissions database from CSV, create default if missing"""
    try:
        if csv_path.exists():
            raw = csv_path.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                df = pd.read_csv(csv_path)
                required = {"name","basis","ef"}
                if required.issubset(set(df.columns)):
                    return df
        fallback_df.to_csv(csv_path, index=False, encoding="utf-8")
        return fallback_df.copy()
    except Exception as e:
        st.warning(f"emissions_food.csv error ({e}); using fallback dataset and recreating file.")
        try:
            fallback_df.to_csv(csv_path, index=False, encoding="utf-8")
        except Exception:
            pass
        return fallback_df.copy()

DB_DF = load_emissions_db(CSV_PATH, FALLBACK_DB)


for col in ["syn","category","alt","alt_basis","alt_note","source"]:
    if col not in DB_DF.columns:
        DB_DF[col] = ""
DB_DF["ef"] = pd.to_numeric(DB_DF["ef"], errors="coerce").fillna(0.0)
DB_DF["syn_list"] = DB_DF["syn"].fillna("").apply(lambda s: [x.strip().lower() for x in str(s).split("|") if x.strip()])

def build_choices(df: pd.DataFrame):
    choices = {}
    for _, r in df.iterrows():
        canon = str(r["name"]).strip()
        s = set([canon.lower()])
        for syn in r.get("syn_list", []):
            s.add(str(syn).lower())
        choices[canon] = list(s)
    return choices

CHOICES = build_choices(DB_DF)

def fuzzy_match_item(line: str, threshold: int = 80):
    q = line.lower().strip()
    best_name, best_score = None, 0
    for canon, strings in CHOICES.items():
        m = process.extractOne(q, strings, scorer=fuzz.token_set_ratio)
        if m and m[1] > best_score:
            best_name, best_score = canon, m[1]
    return (best_name, best_score) if best_score >= threshold else (None, 0)

def parse_receipt_text(text):
    """Parse receipt text and match items to database using fuzzy matching"""
    seen = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.isnumeric():
            continue
        name, score = fuzzy_match_item(line)
        if name:
            seen[name] = max(seen.get(name, 0), int(score))
    rows = []
    for name, score in seen.items():
        row = DB_DF[DB_DF["name"] == name].iloc[0].to_dict()
        default_qty = 1.0
        rows.append({
            "Item": name,
            "Match": score,
            "Basis": row.get("basis","ea"),
            "Qty": default_qty,
            "EF_per_basis (kg CO₂e)": float(row.get("ef", 0) or 0),
            "Source": row.get("source",""),
        })
    df = pd.DataFrame(rows).sort_values("Match", ascending=False)
    if not df.empty:
        df["Emissions (kg CO₂e)"] = (df["Qty"] * df["EF_per_basis (kg CO₂e)"]).round(3)
    return df

def recompute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Qty"] = pd.to_numeric(out["Qty"], errors="coerce").fillna(0.0)
    out["EF_per_basis (kg CO₂e)"] = pd.to_numeric(out["EF_per_basis (kg CO₂e)"], errors="coerce").fillna(0.0)
    out["Emissions (kg CO₂e)"] = (out["Qty"] * out["EF_per_basis (kg CO₂e)"]).round(3)
    return out

def suggest_swaps(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    suggestions = []
    for _, r in df.iterrows():
        item = r["Item"]
        qty = float(r["Qty"])
        db_row = DB_DF[DB_DF["name"] == item]
        if db_row.empty:
            continue
        d = db_row.iloc[0].to_dict()
        alt = str(d.get("alt","")).strip()
        if not alt:
            continue
        # find alt EF from DB if present, else from alt_ef column
        alt_row = DB_DF[DB_DF["name"] == alt]
        if not alt_row.empty:
            alt_ef = float(alt_row.iloc[0].get("ef", 0) or 0)
            alt_note = alt_row.iloc[0].get("alt_note","") or d.get("alt_note","")
        else:
            alt_ef = d.get("alt_ef", None)
            if alt_ef in [None, ""]:
                continue
            alt_ef = float(alt_ef)
            alt_note = d.get("alt_note","")

        current = float(r["EF_per_basis (kg CO₂e)"]) * qty
        alt_em = alt_ef * qty
        savings = round(current - alt_em, 3)
        if savings > 0:
            suggestions.append({
                "Swap": f"{item} → {alt}",
                "Note": alt_note,
                "Savings (kg CO₂e)": savings
            })
    s = pd.DataFrame(suggestions).sort_values("Savings (kg CO₂e)", ascending=False)
    return s.head(top_k) if not s.empty else s

# User interface
tab_text, tab_img, tab_file = st.tabs(["Paste Text", "Upload Image", "Upload File"])
receipt_text = ""

with tab_text:
    receipt_text = st.text_area(
        "Paste receipt text:",
        height=180,
        placeholder="Example:\nMilk 2L\nBread\nBananas\nChicken 1kg",
        key="paste_box"
    )

with tab_img:
    if not TESS_AVAILABLE:
        st.warning("OCR requires Tesseract installation. Use text input instead.")
    else:
        up = st.file_uploader("Upload receipt image", type=["jpg","jpeg","png"], key="img_uploader")
        if up is not None:
            image = Image.open(BytesIO(up.read()))
            st.image(image, caption="Receipt image", use_container_width=True)
            with st.spinner("Extracting text..."):
                try:
                    text = ocr_image(image)
                    st.text_area("Extracted text (editable):", value=text, height=180, key="ocr_box")
                    receipt_text = text
                except Exception as e:
                    st.error(f"OCR failed: {e}")

with tab_file:
    import mimetypes
    file_up = st.file_uploader("Upload receipt file (PDF, TXT, or image)", type=["pdf","txt","jpg","jpeg","png"], key="file_uploader")
    if file_up is not None:
        file_type, _ = mimetypes.guess_type(file_up.name)
        text = ""
        if file_up.name.lower().endswith(".txt"):
            text = file_up.read().decode("utf-8", errors="ignore")
            st.text_area("File content (editable):", value=text, height=180, key="txt_box")
        elif file_up.name.lower().endswith(".pdf"):
            try:
                import fitz  # pdf processing
                pdf_bytes = file_up.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                ocr_text = []
                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    if TESS_AVAILABLE:
                        ocr_text.append(ocr_image(img))
                text = "\n".join(ocr_text)
                st.text_area("PDF text (editable):", value=text, height=180, key="pdf_box")
            except ImportError:
                st.error("Need PyMuPDF for PDF support. Try: pip install pymupdf")
            except Exception as e:
                st.error(f"PDF processing failed: {e}")
        elif file_up.name.lower().endswith((".jpg",".jpeg",".png")):
            try:
                image = Image.open(BytesIO(file_up.read()))
                st.image(image, caption="Uploaded image", use_container_width=True)
                if TESS_AVAILABLE:
                    text = ocr_image(image)
                    st.text_area("Image text (editable):", value=text, height=180, key="imgfile_box")
            except Exception as e:
                st.error(f"Image processing failed: {e}")
        else:
            st.warning("File type not supported.")
        if text:
            receipt_text = text

# Process the receipt and show results
if "items_df" not in st.session_state:
    st.session_state["items_df"] = pd.DataFrame()

if st.button("Process Receipt", type="primary", use_container_width=True):
    if not (receipt_text or "").strip():
        st.error("Please enter receipt text first.")
    else:
        df = parse_receipt_text(receipt_text)
        if df.empty:
            st.warning("No items matched the database. Try editing the text or check item names.")
        else:
            st.success("Items found! Adjust quantities below, then calculate emissions.")
            st.session_state["items_df"] = df

if not st.session_state["items_df"].empty:
    edited = st.data_editor(
        st.session_state["items_df"],
        num_rows="fixed",
        column_config={
            "Item": st.column_config.Column(disabled=True),
            "Match": st.column_config.Column(disabled=True, help="Fuzzy match score (higher is better)"),
            "Basis": st.column_config.SelectboxColumn(options=["kg","L","ea"], disabled=False),
            "Qty": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "EF_per_basis (kg CO₂e)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "Source": st.column_config.Column(disabled=True),
        },
        use_container_width=True,
        key="editor_table",
    )

    if st.button("Calculate Emissions", type="secondary", use_container_width=True):
        final = recompute(edited)
        st.session_state["items_df"] = final  

        total = float(final["Emissions (kg CO₂e)"].sum())
        st.subheader(f"Total carbon footprint: **{total:.2f} kg CO₂e**")

        chart_df = final[["Item","Emissions (kg CO₂e)"]].set_index("Item")
        st.bar_chart(chart_df)

        swaps = suggest_swaps(final, top_k=3)
        if not swaps.empty:
            st.subheader("Suggested alternatives")
            st.dataframe(swaps, use_container_width=True)
        else:
            st.info("No alternative suggestions available.")

        # Save results
        receipts_csv = DATA_DIR / "receipts.csv"
        log = final.copy()
        log.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
        try:
            prev = pd.read_csv(receipts_csv) if receipts_csv.exists() else pd.DataFrame()
            pd.concat([prev, log], ignore_index=True).to_csv(receipts_csv, index=False, encoding="utf-8")
            st.caption("Results saved to data/receipts.csv")
        except Exception:
            st.warning("Could not save results to file.")

        # Download option
        out = final.copy()
        out.loc[len(out.index)] = ["TOTAL","","","", "", total, ""]
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", data=csv_bytes,
                           file_name=f"receipt_{int(time.time())}.csv", mime="text/csv")

st.caption("Edit data/emissions_food.csv to add more items or modify emission factors")
