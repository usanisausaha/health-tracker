
from __future__ import annotations
import streamlit as st
import pandas as pd
import datetime as dt
from typing import Dict, Any

# ==== UI CONFIG (mobile-friendly) ====
st.set_page_config(
    page_title="Health Tracker · Private",
    page_icon="🍎",
    layout="centered",
)

MOBILE_CSS = """
<style>
/* Tighter spacing, bigger inputs for mobile */
.block-container {padding-top: 1rem; padding-bottom: 4rem;}
input, select, textarea {font-size: 1rem !important;}
/* Make dataframes horizontally scrollable on small screens */
div[data-testid="stHorizontalBlock"] {overflow-x: auto;}
/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
"""
st.write(MOBILE_CSS, unsafe_allow_html=True)

#%% ==== CONSTANTS ====
SHEET_NAME_DEFAULT = "Health Tracker"
PROFILE_SHEET = "profile"
FOOD_SHEET = "food_log"
WEIGHT_SHEET = "weight_log"

#%% ==== SECRETS / AUTH ====

# Optional passcode gate
APP_PASSCODE = st.secrets.get("app_passcode", None)

if APP_PASSCODE:
    if "_authed" not in st.session_state:
        st.session_state._authed = False
    if not st.session_state._authed:
        # 👇 เพิ่ม CSS ให้มี margin-top
        st.markdown("""
        <style>
        .block-container {padding-top: 5rem;}
        </style>
        """, unsafe_allow_html=True)

        st.title("🔒 เข้าสู่ระบบ")
        code = st.text_input("รหัสผ่าน", type="password")
        if st.button("เข้าใช้งาน"):
            if code == APP_PASSCODE:
                st.session_state._authed = True
                st.rerun()
            else:
                st.error("รหัสไม่ถูกต้อง")
        st.stop()

#%% ==== GOOGLE SHEETS CLIENT (with local CSV fallback) ====

def get_clients():
    """Return (gspread_client or None, sheet_name)."""
    sheet_name = st.secrets.get("sheet_name", SHEET_NAME_DEFAULT)
    try:
        from google.oauth2.service_account import Credentials
        import gspread
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        sa_info = st.secrets.get("gcp_service_account", None)
        if sa_info is None:
            return None, sheet_name
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        return client, sheet_name
    except Exception as e:
        st.warning(f"ใช้โหมด CSV ชั่วคราว (no Google credentials): {e}")
        return None, sheet_name

client, SHEET_NAME = get_clients()

if client:
    st.sidebar.success(f"✅ Connected to Google Sheet: {SHEET_NAME}")
else:
    st.sidebar.warning("⚠️ Using local CSV fallback (Google Sheet not connected)")

#%% ==== SHEET HELPERS ====

def get_worksheet(client, title: str):
    if client is None:
        return None
    sh = client.open(SHEET_NAME)
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=2000, cols=26)
    return ws

#%% CSV fallbacks (if not using Sheets)
from pathlib import Path
DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
CSV_PROFILE = DATA_DIR / f"{PROFILE_SHEET}.csv"
CSV_FOOD = DATA_DIR / f"{FOOD_SHEET}.csv"
CSV_WEIGHT = DATA_DIR / f"{WEIGHT_SHEET}.csv"

#%% Initialize headers if missing
PROFILE_COLUMNS = [
    "gender", "birthdate", "height_cm", "weight_kg", "body_fat_percent",
    "activity_level", "goal_type", "goal_delta_kcal",
    "protein_g_per_kg", "fat_g_per_kg"
]
FOOD_COLUMNS = [
    "date", "time", "item", "quantity", "unit", "calories",
    "protein_g", "carbs_g", "fat_g", "note"
]
WEIGHT_COLUMNS = ["date", "weight_kg"]


def init_csv_if_needed():
    if not CSV_PROFILE.exists():
        pd.DataFrame(columns=PROFILE_COLUMNS).to_csv(CSV_PROFILE, index=False, encoding="utf-8")
    if not CSV_FOOD.exists():
        pd.DataFrame(columns=FOOD_COLUMNS).to_csv(CSV_FOOD, index=False, encoding="utf-8")
    if not CSV_WEIGHT.exists():
        pd.DataFrame(columns=WEIGHT_COLUMNS).to_csv(CSV_WEIGHT, index=False, encoding="utf-8")


#%% ==== LOAD & SAVE LAYERS ==== 

def load_profile() -> pd.Series:
    if client:
        ws = get_worksheet(client, PROFILE_SHEET)
        values = ws.get_all_values()

        # ถ้า worksheet ว่าง → ใส่ header + sample row
        if not values:
            ws.append_row(PROFILE_COLUMNS)  # set header
            ws.append_row([
                "หญิง", "2000-01-01", 159, 53, "", "นั่งทำงาน",
                "cut", -300, 1.8, 0.8
            ])
            values = ws.get_all_values()

        # ตอนนี้มี header แล้ว → แปลงเป็น DataFrame
        df = pd.DataFrame(values[1:], columns=values[0])

        # ถ้ายังไม่มี row → สร้าง default row
        if df.empty:
            new_row = pd.DataFrame([{
                "gender": "หญิง",
                "birthdate": "2000-01-01",
                "height_cm": 159,
                "weight_kg": 53,
                "body_fat_percent": "",
                "activity_level": "นั่งทำงาน",
                "goal_type": "cut",
                "goal_delta_kcal": -300,
                "protein_g_per_kg": 1.8,
                "fat_g_per_kg": 0.8,
            }])
            ws.append_row(new_row.iloc[0].tolist())
            df = pd.concat([df, new_row], ignore_index=True)

        return df.iloc[0]

    else:
        # --- CSV fallback ---
        init_csv_if_needed()
        df = pd.read_csv(CSV_PROFILE)
        if df.empty:
            df.loc[0, :] = [
                "หญิง", "2000-01-01", 159, 53, "", "นั่งทำงาน",
                "cut", -300, 1.8, 0.8
            ]
            df.to_csv(CSV_PROFILE, index=False)
        return pd.read_csv(CSV_PROFILE).iloc[0]


def save_profile(s: pd.Series):
    if client:
        ws = get_worksheet(client, PROFILE_SHEET)
        ws.clear()
        ws.append_row(PROFILE_COLUMNS)
        ws.append_row([s.get(c, "") for c in PROFILE_COLUMNS])
    else:
        init_csv_if_needed()
        df = pd.DataFrame([s.to_dict()])[PROFILE_COLUMNS]
        df.to_csv(CSV_PROFILE, index=False)


def append_food(row: Dict[str, Any]):
    if client:
        ws = get_worksheet(client, FOOD_SHEET)
        values = ws.get_all_values()
        if not values:
            ws.append_row(FOOD_COLUMNS)
        ws.append_row([row.get(c, "") for c in FOOD_COLUMNS])
    else:
        init_csv_if_needed()
        df = pd.read_csv(CSV_FOOD)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(CSV_FOOD, index=False)

def load_food() -> pd.DataFrame:
    if client:
        ws = get_worksheet(client, FOOD_SHEET)
        values = ws.get_all_values()

        if not values:
            ws.append_row(FOOD_COLUMNS)  # ถ้า sheet ว่าง → ใส่ header
            return pd.DataFrame(columns=FOOD_COLUMNS)

        # Normalize header: lowercase + ตัดช่องว่าง
        raw_headers = [c.strip().lower() for c in values[0]]
        df = pd.DataFrame(values[1:], columns=raw_headers)

        # map ให้ตรงกับ FOOD_COLUMNS
        rename_map = {}
        for col in FOOD_COLUMNS:
            rename_map[col.lower()] = col
        df = df.rename(columns=rename_map)

        # เติมคอลัมน์ที่หายไป
        for col in FOOD_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # แปลงค่าตัวเลขให้เป็น float ถ้าได้
        for col in ["calories","protein_g","carbs_g","fat_g"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df[FOOD_COLUMNS]

    else:
        init_csv_if_needed()
        df = pd.read_csv(CSV_FOOD)

        # บังคับ normalize header ให้เหมือนกัน
        rename_map = {c.lower(): c for c in FOOD_COLUMNS}
        df = df.rename(columns=lambda x: x.strip().lower())
        df = df.rename(columns=rename_map)

        for col in FOOD_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[FOOD_COLUMNS]


def append_weight(row: Dict[str, Any]):
    if client:
        ws = get_worksheet(client, WEIGHT_SHEET)
        values = ws.get_all_values()

        if not values:
            ws.append_row(WEIGHT_COLUMNS)  # header
            values = ws.get_all_values()

        df = pd.DataFrame(values[1:], columns=values[0]) if len(values) > 1 else pd.DataFrame(columns=WEIGHT_COLUMNS)

        # ถ้ามีวันที่นี้แล้ว → อัปเดตแทน
        date_str = row.get("date", "")
        if date_str in df["date"].astype(str).values:
            idx = df.index[df["date"].astype(str) == date_str][0] + 2  # +2 เพราะ index เริ่มที่ 0 และมี header
            ws.update(f"A{idx}:B{idx}", [[row["date"], row["weight_kg"]]])
        else:
            ws.append_row([row.get(c, "") for c in WEIGHT_COLUMNS])

    else:
        init_csv_if_needed()
        df = pd.read_csv(CSV_WEIGHT)

        date_str = row.get("date", "")
        if date_str in df["date"].astype(str).values:
            df.loc[df["date"].astype(str) == date_str, "weight_kg"] = row["weight_kg"]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.to_csv(CSV_WEIGHT, index=False)



def load_weight() -> pd.DataFrame:
    if client:
        ws = get_worksheet(client, WEIGHT_SHEET)
        values = ws.get_all_values()

        if not values:
            ws.append_row(WEIGHT_COLUMNS)
            return pd.DataFrame(columns=WEIGHT_COLUMNS)

        df = pd.DataFrame(values[1:], columns=values[0])

        for col in WEIGHT_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[WEIGHT_COLUMNS]

    else:
        init_csv_if_needed()
        return pd.read_csv(CSV_WEIGHT)


#%% ==== NUTRITION MATH ====
ACTIVITY_MAP = {
    "นั่งทำงาน": 1.2,            # sedentary
    "ออกกำลังน้อย (1-3 วัน/สัปดาห์)": 1.375,
    "ออกกำลังปานกลาง (3-5 วัน/สัปดาห์)": 1.55,
    "ออกกำลังเยอะ (6-7 วัน/สัปดาห์)": 1.725,
    "นักกีฬา/หนักมาก": 1.9,
}

GOALS = {
    "maintain": 0,
    "cut": -300,
    "bulk": 300,
}


def parse_birthdate(s: str) -> int:
    try:
        d = dt.date.fromisoformat(str(s))
        today = dt.date.today()
        age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        return max(0, age)
    except Exception:
        return 25


def bmr_msj(gender: str, weight_kg: float, height_cm: float, age: int) -> float:
    # Mifflin-St Jeor
    if gender.strip() == "ชาย":
        return 10*weight_kg + 6.25*height_cm - 5*age + 5
    else:
        return 10*weight_kg + 6.25*height_cm - 5*age - 161


def bmr_km(weight_kg: float, body_fat_percent: float) -> float:
    # Katch-McArdle (uses lean mass)
    bf = max(0.0, min(60.0, body_fat_percent)) / 100.0
    ffm = weight_kg * (1 - bf)
    return 370 + 21.6 * ffm


def calc_targets(profile: pd.Series) -> Dict[str, Any]:
    gender = profile.get("gender", "หญิง")
    height_cm = float(profile.get("height_cm", 159) or 159)
    weight_kg = float(profile.get("weight_kg", 53) or 53)
    body_fat = profile.get("body_fat_percent", "")
    activity_label = profile.get("activity_level", "นั่งทำงาน")
    goal_type = profile.get("goal_type", "cut")
    goal_delta = float(profile.get("goal_delta_kcal", GOALS.get(goal_type, -300)))
    prot_per_kg = float(profile.get("protein_g_per_kg", 1.8) or 1.8)
    fat_per_kg = float(profile.get("fat_g_per_kg", 0.8) or 0.8)

    age = parse_birthdate(profile.get("birthdate", "2000-01-01"))

    # prefer Katch-McArdle if body fat is provided
    try:
        bf_val = float(body_fat)
        if bf_val > 0:
            bmr = bmr_km(weight_kg, bf_val)
        else:
            bmr = bmr_msj(gender, weight_kg, height_cm, age)
    except Exception:
        bmr = bmr_msj(gender, weight_kg, height_cm, age)

    act = ACTIVITY_MAP.get(activity_label, 1.2)
    tdee = bmr * act
    target_kcal = tdee + goal_delta

    protein_g = max(1.2, prot_per_kg) * weight_kg
    fat_g = max(0.6, fat_per_kg) * weight_kg
    # kcal per macro
    kcal_from_protein = protein_g * 4
    kcal_from_fat = fat_g * 9
    carbs_kcal = max(0.0, target_kcal - kcal_from_protein - kcal_from_fat)
    carbs_g = carbs_kcal / 4

    return {
        "age": age,
        "bmr": bmr,
        "tdee": tdee,
        "target_kcal": target_kcal,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
    }


#%% ==== APP UI ====

st.title("🍎 Health Tracker (Private)")

profile = load_profile()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["โปรไฟล์ & เป้าหมาย", "บันทึกอาหารวันนี้", "สถิติ/สรุป"])

with tab1:
    st.subheader("📋 โปรไฟล์")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("เพศ", ["หญิง", "ชาย"], index=0 if profile.get("gender","หญิง")=="หญิง" else 1)
        birthdate = st.date_input("วันเกิด", value=dt.date.fromisoformat(str(profile.get("birthdate","2000-01-01"))))
        height_cm = st.number_input("ส่วนสูง (ซม.)", min_value=100, max_value=220, value=int(float(profile.get("height_cm", 159) or 159)))
        weight_kg = st.number_input("น้ำหนักปัจจุบัน (กก.)", min_value=30.0, max_value=200.0, value=float(profile.get("weight_kg", 53) or 53.0))
    with col2:
        body_fat = st.text_input("เปอร์เซ็นต์ไขมัน % (ถ้าทราบ)", value=str(profile.get("body_fat_percent", "")))
        activity = st.selectbox("ระดับกิจกรรม", list(ACTIVITY_MAP.keys()), index=list(ACTIVITY_MAP.keys()).index(profile.get("activity_level","นั่งทำงาน")))
        goal_type = st.radio("เป้าหมาย", ["maintain","cut","bulk"], index=["maintain","cut","bulk"].index(profile.get("goal_type","cut")))
        goal_delta = st.number_input("ปรับแคลอรี (เช่น -300 สำหรับลดน้ำหนัก)", -1500, 1500, value=int(float(profile.get("goal_delta_kcal", GOALS.get(goal_type, -300)))))

    st.markdown("**สัดส่วนแมโคร (แนะนำ)** — ปรับได้ตามชอบ")
    c1, c2 = st.columns(2)
    with c1:
        protein_per_kg = st.number_input("โปรตีน (กรัม/กก.)", 0.8, 3.0, value=float(profile.get("protein_g_per_kg", 1.8)), step=0.1)
    with c2:
        fat_per_kg = st.number_input("ไขมัน (กรัม/กก.)", 0.4, 1.5, value=float(profile.get("fat_g_per_kg", 0.8)), step=0.1)

    # Save button
    if st.button("บันทึกโปรไฟล์/เป้าหมาย"):
        new_profile = pd.Series({
            "gender": gender,
            "birthdate": birthdate.isoformat(),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "body_fat_percent": body_fat,
            "activity_level": activity,
            "goal_type": goal_type,
            "goal_delta_kcal": goal_delta,
            "protein_g_per_kg": protein_per_kg,
            "fat_g_per_kg": fat_per_kg,
        })
        save_profile(new_profile)
        st.success("บันทึกแล้ว ✅")
        st.rerun()

    # Calculations display
    targets = calc_targets(load_profile())
    st.subheader("📐 ค่าที่คำนวณได้")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("อายุ (ปี)", f"{targets['age']}")
    m2.metric("BMR", f"{targets['bmr']:.0f} kcal")
    m3.metric("TDEE", f"{targets['tdee']:.0f} kcal")
    m4.metric("เป้า/วัน", f"{targets['target_kcal']:.0f} kcal")

    st.markdown("**เป้าแมโคร/วัน**")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("โปรตีน", f"{targets['protein_g']:.0f} g")
    mc2.metric("คาร์บ", f"{targets['carbs_g']:.0f} g")
    mc3.metric("ไขมัน", f"{targets['fat_g']:.0f} g")

with tab2:
    st.subheader("🍽️ บันทึกอาหารวันนี้")

    today = st.date_input("วันที่", dt.date.today())
    t = dt.datetime.now().strftime("%H:%M")

    with st.form("food_form"):
        colA, colB = st.columns([2,1])
        with colA:
            item = st.text_input("เมนู/อาหาร")
            note = st.text_input("หมายเหตุ")
        with colB:
            quantity = st.number_input("จำนวน", min_value=0.0, value=1.0, step=0.5)
            unit = st.text_input("หน่วย (เช่น จาน, กรัม)", value="จาน")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            calories = st.number_input("kcal", min_value=0.0, value=0.0, step=10.0)
        with c2:
            protein_g = st.number_input("โปรตีน (g)", min_value=0.0, value=0.0, step=1.0)
        with c3:
            carbs_g = st.number_input("คาร์บ (g)", min_value=0.0, value=0.0, step=1.0)
        with c4:
            fat_g = st.number_input("ไขมัน (g)", min_value=0.0, value=0.0, step=1.0)

        submitted = st.form_submit_button("บันทึกเมนู")
        if submitted:
            if not item:
                st.error("กรอกชื่อเมนูด้วยนะ")
            else:
                append_food({
                    "date": today.isoformat(),
                    "time": t,
                    "item": item,
                    "quantity": quantity,
                    "unit": unit,
                    "calories": calories,
                    "protein_g": protein_g,
                    "carbs_g": carbs_g,
                    "fat_g": fat_g,
                    "note": note,
                })
                st.success("บันทึกแล้ว ✅")
                st.experimental_set_query_params(_=dt.datetime.now().timestamp())  # trick to refresh
                st.rerun()

    # Today summary
    df_food = load_food()
    if not df_food.empty:
        df_food["date"] = pd.to_datetime(df_food["date"]).dt.date
        df_today = df_food[df_food["date"] == today]
        st.markdown("### สรุปวันนี้")
        if df_today.empty:
            st.info("ยังไม่มีรายการอาหารวันนี้")
        else:
            totals = df_today[["calories","protein_g","carbs_g","fat_g"]].sum(numeric_only=True)
            tg = calc_targets(load_profile())
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("kcal รวม", f"{totals['calories']:.0f}", delta=f"เป้า {tg['target_kcal']:.0f}")
            cc2.metric("โปรตีน", f"{totals['protein_g']:.0f} g", delta=f"เป้า {tg['protein_g']:.0f} g")
            cc3.metric("คาร์บ", f"{totals['carbs_g']:.0f} g", delta=f"เป้า {tg['carbs_g']:.0f} g")
            cc4.metric("ไขมัน", f"{totals['fat_g']:.0f} g", delta=f"เป้า {tg['fat_g']:.0f} g")

            st.dataframe(
                df_today.sort_values(["time"]).reset_index(drop=True),
                use_container_width=True,
            )

with tab3:
    st.subheader("📈 สถิติ & น้ำหนัก")

    # Weight log form
    wcol1, wcol2 = st.columns([1,1])
    with wcol1:
        w_date = st.date_input("วันที่ชั่ง", dt.date.today(), key="wdate")
    with wcol2:
        w_val = st.number_input("น้ำหนัก (กก.)", min_value=30.0, max_value=200.0, step=0.1)
    if st.button("บันทึกน้ำหนัก"):
        append_weight({"date": w_date.isoformat(), "weight_kg": float(w_val)})
        st.success("บันทึกแล้ว ✅")
        st.rerun()

    df_w = load_weight()
    if not df_w.empty:
        df_w["date"] = pd.to_datetime(df_w["date"]).dt.date
        df_w = df_w.sort_values("date")
        st.markdown("**กราฟน้ำหนัก**")
        st.line_chart(df_w.set_index("date")["weight_kg"])  # simple chart
        st.dataframe(df_w.rename(columns={"date":"วันที่","weight_kg":"น้ำหนัก (กก.)"}), use_container_width=True)

    # Calorie trend
    df_food = load_food()
    if not df_food.empty:
        df_food["date"] = pd.to_datetime(df_food["date"]).dt.date
        daily = df_food.groupby("date").agg({"calories":"sum","protein_g":"sum","carbs_g":"sum","fat_g":"sum"}).reset_index()
        st.markdown("**แคลอรีต่อวัน**")
        st.bar_chart(daily.set_index("date")["calories"])  # bar chart
        st.dataframe(daily.rename(columns={"date":"วันที่","calories":"kcal"}), use_container_width=True)

st.caption("ทำงานได้ทั้งบนมือถือและเดสก์ท็อป · เก็บข้อมูลบน Google Sheets หรือ CSV ในเครื่องถ้าไม่มี credentials")

# ====== END OF FILE ======
