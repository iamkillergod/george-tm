"""
George TM Scanner — Streamlit app

Usage:
1. Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows

2. Install requirements:
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

The app accepts .xlsx/.xls files and outputs a similarity report (downloadable).
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
import re
import io
import math
from typing import Optional, Tuple

st.set_page_config(page_title="George TM Scanner", layout="wide")

# ---------- Helper functions ----------

def clean_text(s: object) -> str:
    """Lowercase, remove punctuation except spaces, collapse whitespace."""
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def seq_ratio(a: str, b: str) -> float:
    """Character-level sequence similarity (0..1)."""
    return SequenceMatcher(None, a, b).ratio()

def token_sort_ratio(a: str, b: str) -> float:
    """Token-sort ratio: sort words then compare sequence ratio."""
    ta = " ".join(sorted(a.split()))
    tb = " ".join(sorted(b.split()))
    return SequenceMatcher(None, ta, tb).ratio()

def soundex(s: str) -> str:
    """Simple Soundex implementation (4-character code)."""
    s = re.sub(r"[^A-Za-z]", "", (s or "").upper())
    if not s:
        return ""
    # mapping groups
    codes = {
        "BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6"
    }
    def char_code(ch: str) -> str:
        for k, v in codes.items():
            if ch in k:
                return v
        return ""
    # first letter
    out = s[0]
    last = char_code(out)
    for ch in s[1:]:
        c = char_code(ch)
        if c != last and c != "":
            out += c
        last = c or last
    out = (out + "000")[:4]
    return out

def phonetic_similarity(a: str, b: str) -> float:
    sa = soundex(a)
    sb = soundex(b)
    return 1.0 if sa and sb and sa == sb else 0.0

def compute_similarity(a_raw: object, b_raw: object) -> Optional[Tuple[float, float, float, float]]:
    """Return (combined_pct, seq, token_sort, phonetic_flag) or None for empty pair."""
    a = clean_text(a_raw)
    b = clean_text(b_raw)
    if a == "" and b == "":
        return None
    s1 = seq_ratio(a, b)
    s2 = token_sort_ratio(a, b)
    s3 = phonetic_similarity(a, b)
    combined = 0.45 * s1 + 0.35 * s2 + 0.20 * s3
    return round(combined * 100, 2), round(s1, 4), round(s2, 4), int(s3)

def find_header_row_from_preview(df_raw: pd.DataFrame) -> int:
    """Try to find header row index by scanning first 10 rows for 'applicant' and 'opponent' words."""
    for i in range(min(10, len(df_raw))):
        row_text = " ".join([str(x).lower() for x in df_raw.iloc[i].fillna("")])
        if "applicant" in row_text and "opponent" in row_text:
            return i
    return 0  # default to first row if not found

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect Applicant and Opponent column names in a dataframe (already header row applied)."""
    def find_col_by_keywords(keywords):
        for col in df.columns:
            lc = str(col).lower()
            for kw in keywords:
                if kw in lc:
                    return col
        return None

    app_col = find_col_by_keywords(["applicant trademark", "applicant mark", "applicant", "applicant_name", "applicantname"])
    opp_col = find_col_by_keywords(["opponent trademark", "opponent mark", "opponent", "opponent_name", "opponentname"])
    # broader fallbacks
    if app_col is None:
        for col in df.columns:
            if "applicant" in str(col).lower() or "app" in str(col).lower():
                app_col = col
                break
    if opp_col is None:
        for col in df.columns:
            if "opponent" in str(col).lower() or "opp" in str(col).lower():
                opp_col = col
                break
    return app_col, opp_col

# ---------- Streamlit UI ----------

st.title("George TM Scanner")
st.write("Upload an Excel trademark report. The app auto-detects Applicant/Opponent and computes an AI-style similarity score (sequence + token-sort + phonetic).")

uploaded = st.file_uploader("Upload Excel file (.xlsx/.xls)", type=["xlsx", "xls"])
priority_txt = st.text_input("Priority client keywords (comma-separated)", value="vguard,lulu,aster,kannan,kannandevan,ved")
priority_keywords = [k.strip().lower() for k in priority_txt.split(",") if k.strip()]
threshold = st.slider("Similarity threshold for FLAG (≥)", 0, 100, 60)

process_btn = st.button("Scan file")

if uploaded is None:
    st.info("Upload a file to get started. If detection fails, ensure the sheet's header row contains the words 'Applicant' and 'Opponent'.")
else:
    try:
        # Read raw with no header to detect header row
        raw_preview = pd.read_excel(uploaded, header=None)
        header_row = find_header_row_from_preview(raw_preview)
        uploaded.seek(0)
        df = pd.read_excel(uploaded, header=header_row)
        # normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        app_col, opp_col = detect_columns(df)

        if app_col is None or opp_col is None:
            st.error("Could not detect Applicant or Opponent columns automatically. Please ensure headers contain 'Applicant' and 'Opponent', or specify them manually in the sheet.")
            st.caption(f"Detected columns: {list(df.columns)[:10]} ...")
        else:
            st.success(f"Detected Applicant column: **{app_col}**  |  Opponent column: **{opp_col}**")

            if process_btn:
                with st.spinner("Computing similarity scores..."):
                    rows = []
                    for idx, row in df.iterrows():
                        a_raw = row.get(app_col, "")
                        b_raw = row.get(opp_col, "")
                        sim = compute_similarity(a_raw, b_raw)
                        if sim is None:
                            continue
                        combined, s_seq, s_token, s_phon = sim
                        # normalized simple token for priority matching
                        app_norm = re.sub(r"[^a-z0-9]", "", str(a_raw).lower())
                        opp_norm = re.sub(r"[^a-z0-9]", "", str(b_raw).lower())
                        is_priority = any(k in app_norm or k in opp_norm for k in priority_keywords)
                        rows.append({
                            "Original Row": idx + 1,
                            "Applicant Raw": a_raw,
                            "Opponent Raw": b_raw,
                            "SeqRatio": s_seq,
                            "TokenSort": s_token,
                            "PhoneticMatch": s_phon,
                            "AI_Similarity_%": combined,
                            "PriorityClient": "Yes" if is_priority else "No"
                        })
                    if not rows:
                        st.warning("No valid Applicant/Opponent pairs found after header row.")
                    else:
                        result_df = pd.DataFrame(rows).sort_values("AI_Similarity_%", ascending=False).reset_index(drop=True)
                        st.markdown("### Summary")
                        st.write(f"Total comparisons: **{len(result_df)}**")
                        flagged = result_df[result_df["AI_Similarity_%"] >= threshold]
                        st.write(f"Flags (Similarity ≥ {threshold}%): **{len(flagged)}**")
                        st.write("Top flagged rows (first 200):")
                        st.dataframe(flagged.head(200), use_container_width=True)

                        st.markdown("### Priority client hits")
                        pri_hits = result_df[result_df["PriorityClient"] == "Yes"]
                        st.write(f"Priority hits: **{len(pri_hits)}**")
                        st.dataframe(pri_hits.head(200), use_container_width=True)

                        # Full interactive table (collapsible)
                        with st.expander("Show full scored table (first 1000 rows)"):
                            st.dataframe(result_df.head(1000), use_container_width=True)

                        # Offer downloads: CSV and Excel
                        csv_buf = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV", csv_buf, "George_TM_Similarity_Report.csv", mime="text/csv")

                        # Excel writer
                        to_download = result_df.copy()
                        # try merge back other columns for context (safe)
                        try:
                            extras = [c for c in df.columns if c not in [app_col, opp_col]]
                            merged = df[[app_col, opp_col] + extras].reset_index().rename(columns={"index": "Original Row"})
                            to_download = pd.merge(result_df, merged, on="Original Row", how="left")
                        except Exception:
                            pass
                        out = io.BytesIO()
                        with pd.ExcelWriter(out, engine="openpyxl") as writer:
                            to_download.to_excel(writer, index=False, sheet_name="Similarity_Report")
                        st.download_button("Download Excel", out.getvalue(), "George_TM_Similarity_Report.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Error reading file or processing: {e}")
        st.exception(e)
