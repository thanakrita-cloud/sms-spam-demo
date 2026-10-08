"""SMS Spam Checker (English) -- เดโมของโปรเจกต์ CP020003
รัน:  python -m streamlit run app.py     หรือดับเบิลคลิก run_app.bat
โหลดโมเดลที่เทรนไว้แล้วจาก model/models.joblib (สร้างด้วย build_model.py) ไม่เทรนใหม่ตอนกดตรวจ
"""
import html
import json
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scope_rules import HARD_LIMIT, MAX_CHARS, scope_issue

HERE = Path(__file__).parent
SCALE = 12.0              # แถบคะแนนแสดงช่วง -12 ถึง +12 (เกินนี้ยังตัดสินได้ แค่แถบเต็ม)
TOKEN = re.compile(r'(?u)\b\w\w+\b')   # เหมือน token ของ TfidfVectorizer

EXAMPLES = [
    ('spam', '🚨 ถูกรางวัล', 'URGENT! You have won a £1000 cash prize. Call 09061234567 now to claim your reward.'),
    ('spam', '🚨 ชวนชิงรางวัล', 'Free entry in a weekly competition to win tickets! Text WIN to 80085 now. T&Cs apply.'),
    ('ham', '💬 นัดกินข้าว', 'Hey, are we still meeting for lunch tomorrow? Let me know.'),
    ('ham', '💬 ฝากซื้อของ', "I'll be home by 7. Can you pick up some milk on the way?"),
    ('hard', '⚠️ ปกติ แต่โมเดลว่า spam', 'Waiting for your call.'),
    ('hard', '⚠️ spam แต่โมเดลว่าปกติ', 'For sale - arsenal dartboard. Good condition but no doubles or trebles!'),
]

CSS = """
<style>
html, body, [class*="css"], .stMarkdown, button, textarea { font-family: 'Leelawadee UI','Tahoma','Segoe UI',sans-serif; }
.block-container { max-width: 980px; padding-top: 4.4rem; padding-bottom: 3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }

/* ---------- hero banner ---------- */
.hero { display:flex; gap:18px; align-items:center; margin-bottom:.9rem; padding:22px 26px; border-radius:20px;
        background:linear-gradient(135deg,#1E3A8A 0%,#2F6FDE 62%,#4F8DF5 100%); box-shadow:0 8px 22px rgba(30,58,138,.28); }
.hero > div:last-child { min-width:0; }
.hero-icon { width:62px; height:62px; border-radius:16px; background:#FFFFFF; display:flex; align-items:center; justify-content:center; font-size:32px; flex:none; box-shadow:0 2px 8px rgba(0,0,0,.2); }
.hero h1, .hero h1 span { margin:0; padding:0; font-size:2rem; line-height:1.2; color:#FFFFFF !important; }
.hero p { margin:.35rem 0 0; color:#E3ECFF !important; opacity:1; }
.hero h1 span.badge { display:inline-block; font-size:.74rem !important; line-height:1.5 !important; font-weight:700; padding:.12rem .75rem; border-radius:999px; background:rgba(255,255,255,.18); color:#FFFFFF !important; border:1px solid rgba(255,255,255,.65); margin-left:.7rem; vertical-align:middle; }
.steps { display:flex; gap:22px; flex-wrap:wrap; margin:.2rem 0 .6rem; }
.step { display:flex; align-items:center; gap:9px; font-size:.95rem; font-weight:600; }
.step b { width:28px; height:28px; border-radius:50%; background:#2563EB; color:#fff; display:inline-flex; align-items:center; justify-content:center; font-size:.85rem; box-shadow:0 2px 6px rgba(37,99,235,.45); }
.step:nth-child(2) b { background:#EA580C; box-shadow:0 2px 6px rgba(234,88,12,.45); }
.step:nth-child(3) b { background:#7C3AED; box-shadow:0 2px 6px rgba(124,58,237,.45); }

/* ---------- headings ---------- */
.section-label { font-weight:800; font-size:1.02rem; margin:1.3rem 0 .5rem; padding-left:11px; border-left:5px solid #2563EB; line-height:1.2; }
.group-label { font-size:.85rem; font-weight:800; margin-bottom:.25rem; letter-spacing:.2px; }
.group-label.sp { color:#C2410C; } .group-label.hm { color:#1D4ED8; } .group-label.hd { color:#B45309; }
button[data-baseweb="tab"] { font-weight:700; }

/* ---------- example buttons (แยกสีตามกลุ่ม) ---------- */
.st-key-ex0 button, .st-key-ex1 button { background:#FFEDD5 !important; border:2px solid #EA580C !important; color:#9A3412 !important; font-weight:700 !important; }
.st-key-ex0 button:hover, .st-key-ex1 button:hover { background:#C2410C !important; color:#FFFFFF !important; }
.st-key-ex2 button, .st-key-ex3 button { background:#DBEAFE !important; border:2px solid #2563EB !important; color:#1E3A8A !important; font-weight:700 !important; }
.st-key-ex2 button:hover, .st-key-ex3 button:hover { background:#1D4ED8 !important; color:#FFFFFF !important; }
.st-key-ex4 button, .st-key-ex5 button { background:#FEF3C7 !important; border:2px dashed #D97706 !important; color:#78350F !important; font-weight:700 !important; }
.st-key-ex4 button:hover, .st-key-ex5 button:hover { background:#B45309 !important; color:#FFFFFF !important; }
div.stButton > button { border-radius:12px; }

/* ---------- input ---------- */
div[data-baseweb="textarea"] { border:2px solid #93B4F0 !important; border-radius:14px !important; background:#FFFFFF !important; }
div[data-baseweb="textarea"]:focus-within { border-color:#2563EB !important; box-shadow:0 0 0 4px rgba(37,99,235,.22); }
textarea { font-size:1.05rem !important; color:#1B2433 !important; background:#FFFFFF !important; }
button[data-testid="stBaseButton-primaryFormSubmit"] { background:linear-gradient(135deg,#2563EB,#4F46E5) !important; color:#FFFFFF !important; border:none !important; border-radius:12px; font-weight:800 !important; font-size:1.05rem !important; padding:.6rem 1rem !important; box-shadow:0 5px 14px rgba(37,99,235,.42); }
button[data-testid="stBaseButton-primaryFormSubmit"]:hover { filter:brightness(1.1); }
button[data-testid="stBaseButton-secondaryFormSubmit"] { border:2px solid #94A3B8 !important; border-radius:12px; font-weight:700 !important; }

/* ---------- result card ---------- */
.result { border-radius:20px; padding:24px 28px 18px; margin:6px 0 12px; color:#FFFFFF; }
.result.spam { background:linear-gradient(135deg,#D4470A 0%,#B53A07 100%); box-shadow:0 10px 26px rgba(212,71,10,.38); }
.result.ham  { background:linear-gradient(135deg,#2563EB 0%,#1E40AF 100%); box-shadow:0 10px 26px rgba(37,99,235,.38); }
.result.unknown { background:linear-gradient(135deg,#64748B 0%,#475569 100%); box-shadow:0 10px 26px rgba(71,85,105,.30); }
.result .big { font-size:2.1rem; font-weight:800; line-height:1.2; color:#FFFFFF; }
.result .sub { margin-top:8px; font-size:1.02rem; color:#FFFFFF; }
.result .sub b { font-weight:800; text-decoration:underline; text-underline-offset:3px; }

.gauge { position:relative; height:18px; border-radius:999px; margin:36px 10px 8px; border:2px solid rgba(255,255,255,.85);
         background:linear-gradient(90deg,#1D4ED8 0%,#60A5FA 28%,#E0ECFF 47%,#FFE8CC 53%,#FB923C 75%,#C2410C 100%); box-shadow:inset 0 1px 3px rgba(0,0,0,.25); }
.gauge .zero { position:absolute; left:50%; top:-8px; width:3px; height:30px; background:#FFFFFF; border-radius:2px; box-shadow:0 0 0 1px rgba(0,0,0,.25); }
.gauge .pin { position:absolute; top:-12px; transform:translateX(-50%); min-width:62px; height:34px; padding:0 12px; border-radius:17px; background:#FFFFFF; color:#1B2433;
              font-size:.95rem; font-weight:800; display:flex; align-items:center; justify-content:center; box-shadow:0 3px 8px rgba(0,0,0,.35); }
.gauge-labels { display:flex; justify-content:space-between; font-size:.82rem; font-weight:600; color:#FFFFFF; opacity:.95; margin:0 8px 4px; }

/* ---------- message + highlights ---------- */
.msg { background:#FFFFFF; color:#1B2433; border:2px solid #CBD5E1; border-radius:14px; padding:14px 18px; line-height:2; word-break:break-word; font-size:1.03rem; }
mark.sp { background:#FDBA74; color:#7C2D12; border-radius:6px; padding:2px 6px; font-weight:700; }
mark.hm { background:#93C5FD; color:#1E3A8A; border-radius:6px; padding:2px 6px; font-weight:700; }
.legend { font-size:.85rem; margin-top:8px; }
.legend mark { margin-right:4px; }

/* ---------- model chips ---------- */
.chips { display:flex; gap:10px; flex-wrap:wrap; margin:.2rem 0 .3rem; }
.chip { background:#FFFFFF; color:#1B2433; border:2px solid #CBD5E1; border-radius:14px; padding:8px 14px; font-size:.92rem; font-weight:600; }
.chip b { padding:2px 10px; border-radius:999px; color:#FFFFFF; margin-left:4px; }
.chip b.s { background:#C2410C; } .chip b.h { background:#1D4ED8; }
.chip.ok   { background:#DCFCE7; border-color:#4ADE80; color:#14532D; }
.chip.warn { background:#FEF3C7; border-color:#F59E0B; color:#78350F; }
.note { font-size:.86rem; opacity:.8; margin-top:.5rem; }

/* ---------- about tab ---------- */
div[data-testid="stMetric"] { background:#FFFFFF; border:2px solid #BFDBFE; border-radius:16px; padding:14px 18px; box-shadow:0 3px 10px rgba(37,99,235,.12); }
div[data-testid="stMetric"] * { color:#1B2433 !important; }

@media (max-width: 640px) {
  .hero { padding:16px 16px; gap:12px; }
  .hero h1, .hero h1 span { font-size:1.45rem !important; }
  .hero h1 span.badge { font-size:.72rem !important; margin-left:.4rem; }
  .hero-icon { width:48px !important; height:48px !important; font-size:26px !important; }
  .result .big { font-size:1.6rem; }
  .steps { gap:10px 16px; }
}
</style>
"""


@st.cache_resource
def load():
    path = HERE / 'model' / 'models.joblib'
    if not path.exists():
        return None, None, None
    models = joblib.load(path)
    meta = json.load(open(HERE / 'model' / 'meta.json', encoding='utf-8'))
    names = models['LogReg'].steps[0][1].get_feature_names_out()
    return models, meta, names


def explain(pipe, names, text, k=8):
    """แยกคะแนนของ Logistic Regression เป็นส่วนของแต่ละคำ/คู่คำ (+ = ดันไป spam, - = ดันไป ham)"""
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    x = vec.transform([text]).tocsr()
    contrib = x.data * clf.coef_[0][x.indices]
    intercept = float(clf.intercept_[0])
    score = float(contrib.sum() + intercept)
    order = np.argsort(-np.abs(contrib))[:k]
    rows = [(str(names[x.indices[i]]), float(contrib[i])) for i in order]
    return score, intercept, rows, int(x.nnz)


def highlight(text, rows):
    """ไฮไลต์คำในข้อความที่โมเดลให้น้ำหนักมาก (escape ทุกส่วนก่อนใส่ HTML)"""
    weight = {}
    for term, c in rows:
        for tok in term.split():
            if abs(c) > abs(weight.get(tok, 0)):
                weight[tok] = c
    shown = text if len(text) <= 700 else text[:700] + ' …'
    out, last = [], 0
    for m in TOKEN.finditer(shown):
        out.append(html.escape(shown[last:m.start()]))
        seg = html.escape(m.group(0))
        w = weight.get(m.group(0).lower())
        out.append(f'<mark class="{"sp" if w > 0 else "hm"}">{seg}</mark>' if w is not None else seg)
        last = m.end()
    out.append(html.escape(shown[last:]))
    return ''.join(out).replace('\n', '<br>')


def gauge(score):
    pct = (max(-SCALE, min(SCALE, score)) + SCALE) / (2 * SCALE) * 100
    pin = max(9, min(91, pct))     # ไม่ให้ป้ายล้นขอบ
    return (f'<div class="gauge" role="img" aria-label="คะแนน {score:+.2f} บนสเกล -{SCALE:.0f} ถึง +{SCALE:.0f}">'
            f'<div class="zero"></div><div class="pin" style="left:{pin:.1f}%">{score:+.2f}</div></div>'
            f'<div class="gauge-labels"><span>← ham</span><span>เส้นตัดสิน 0</span><span>spam →</span></div>')


# ---------------------------------------------------------------- page
st.set_page_config(page_title='SMS Spam Checker', page_icon='✉️', layout='centered', initial_sidebar_state='collapsed')
st.markdown(CSS, unsafe_allow_html=True)
models, meta, names = load()

st.markdown(
    '<div class="hero"><div class="hero-icon">✉️</div><div>'
    '<h1>SMS Spam Checker<span class="badge">English only</span></h1>'
    '<p>วางข้อความ SMS ภาษาอังกฤษ แล้วดูว่าโมเดลมองว่าเป็น spam หรือข้อความปกติ พร้อมคำที่ทำให้โมเดลคิดแบบนั้น</p>'
    '</div></div>'
    '<div class="steps"><div class="step"><b>1</b>วางข้อความ หรือกดตัวอย่าง</div>'
    '<div class="step"><b>2</b>กด "ตรวจข้อความ"</div><div class="step"><b>3</b>อ่านผลและคำที่มีผล</div></div>',
    unsafe_allow_html=True)

if models is None:
    st.error('ยังไม่มีไฟล์โมเดล รัน `python build_model.py` ในโฟลเดอร์ app ก่อน')
    st.stop()

if 'sms_text' not in st.session_state:
    st.session_state.sms_text = ''
if 'checked' not in st.session_state:
    st.session_state.checked = None


def use_example(text):
    st.session_state.sms_text = text
    st.session_state.checked = text


def do_check():
    st.session_state.checked = st.session_state.sms_text


def do_clear():
    st.session_state.sms_text = ''
    st.session_state.checked = None


tab_check, tab_about = st.tabs(['ตรวจข้อความ', 'เกี่ยวกับโมเดลและข้อจำกัด'])

with tab_check:
    st.markdown('<div class="section-label">ลองตัวอย่างก่อนก็ได้</div>', unsafe_allow_html=True)
    groups = [('spam', 'sp', '🟠 Spam'), ('ham', 'hm', '🔵 ข้อความปกติ'), ('hard', 'hd', '🟡 ที่โมเดลทายพลาด (ดูข้อจำกัด)')]
    for gcol, (kind, gcls, gname) in zip(st.columns(3), groups):
        with gcol:
            st.markdown(f'<div class="group-label {gcls}">{gname}</div>', unsafe_allow_html=True)
            for i, (k, label, text) in enumerate(EXAMPLES):
                if k == kind:
                    st.button(label, key=f'ex{i}', on_click=use_example, args=(text,), width='stretch')

    st.markdown('<div class="section-label">หรือวางข้อความของคุณ</div>', unsafe_allow_html=True)
    with st.form('check_form', border=False):
        st.text_area('ข้อความ SMS', key='sms_text', height=130, label_visibility='collapsed',
                     placeholder='เช่น: Congratulations! You have won a free prize. Call now to claim...')
        b1, b2 = st.columns([3, 1])
        b1.form_submit_button('🔍 ตรวจข้อความ', type='primary', on_click=do_check, width='stretch')
        b2.form_submit_button('ล้าง', on_click=do_clear, width='stretch')
        st.caption('กด Ctrl + Enter เพื่อตรวจได้เลย')

    text = st.session_state.sms_text
    checked = st.session_state.checked

    if checked is not None and not checked.strip():
        st.warning('ยังไม่มีข้อความ ลองวางข้อความหรือกดตัวอย่างด้านบน')
    elif checked is not None:
        if checked != text:
            st.caption('⚠️ ข้อความในช่องเปลี่ยนไปแล้ว ผลด้านล่างเป็นของข้อความก่อนหน้า กด "ตรวจข้อความ" เพื่ออัปเดต')

        query = checked.strip()
        issue = scope_issue(query)
        score = intercept = None
        rows, n_known = [], 0
        if issue is None:
            score, intercept, rows, n_known = explain(models['LogReg'], names, query)
            issue = scope_issue(query, n_known)

        st.markdown('<div class="section-label">ผลการตรวจ</div>', unsafe_allow_html=True)
        if issue is not None:
            st.markdown(
                '<div class="result unknown"><div class="big">⚪ ประเมินไม่ได้</div>'
                f'<div class="sub">{html.escape(issue)}</div></div>',
                unsafe_allow_html=True)
            st.info('ลองใช้ SMS ภาษาอังกฤษแบบสั้นที่มีคำทั่วไป ระบบจะไม่แสดง HAM/SPAM เมื่อข้อมูลอยู่นอกขอบเขต')
        else:
            if len(query) > MAX_CHARS:
                st.warning(f'ข้อความยาวผิดปกติ ({len(query):,} ตัวอักษร) ระบบออกแบบสำหรับ SMS สั้น ผลอาจไม่น่าเชื่อถือ')

            is_spam = score > 0
            lab = {0: 'ham', 1: 'spam'}
            others = {n: int(models[n].predict([query])[0]) for n in ['NB', 'SVM']}
            pos = [t for t, c in rows if c > 0][:3]
            neg = [t for t, c in rows if c < 0][:3]
            if is_spam:
                why = ('คำที่ดันไปทาง spam มากสุด: ' + ', '.join(f'<b>{html.escape(t)}</b>' for t in pos)) if pos else 'ไม่มีคำเด่นชัดเจน'
                card = f'<div class="result spam"><div class="big">🚨 น่าจะเป็น SPAM</div><div class="sub">{why}</div>{gauge(score)}</div>'
            else:
                why = ('คำที่ดันไปทาง ham มากสุด: ' + ', '.join(f'<b>{html.escape(t)}</b>' for t in neg)) if neg else 'ไม่เจอคำที่บ่งชี้ว่าเป็น spam'
                card = f'<div class="result ham"><div class="big">✅ น่าจะเป็นข้อความปกติ (HAM)</div><div class="sub">{why}</div>{gauge(score)}</div>'
            st.markdown(card, unsafe_allow_html=True)
            if abs(score) < 1:
                st.warning('คะแนนใกล้ 0 โมเดลคาบเส้น ผลนี้เชื่อถือได้น้อย')

            agree = len({others['NB'], others['SVM'], int(is_spam)}) == 1
            st.markdown(
                '<div class="chips">'
                f'<div class="chip">Logistic Regression: <b class="{"s" if is_spam else "h"}">{lab[int(is_spam)]}</b></div>'
                f'<div class="chip">Naive Bayes: <b class="{"s" if others["NB"] else "h"}">{lab[others["NB"]]}</b></div>'
                f'<div class="chip">Linear SVM: <b class="{"s" if others["SVM"] else "h"}">{lab[others["SVM"]]}</b></div>'
                f'<div class="chip {"ok" if agree else "warn"}">{"✔ ทั้งสามโมเดลเห็นตรงกัน" if agree else "⚠ โมเดลเห็นไม่ตรงกัน ควรระวัง"}</div></div>',
                unsafe_allow_html=True)

            st.markdown('<div class="section-label">คำในข้อความที่โมเดลให้น้ำหนัก</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg">{highlight(query, rows)}</div>'
                        '<div class="legend"><mark class="sp">ส้ม</mark>ดันไปทาง spam &nbsp; <mark class="hm">ฟ้า</mark>ดันไปทาง ham</div>',
                        unsafe_allow_html=True)
            df = pd.DataFrame(rows, columns=['term', 'contribution'])
            df['ทิศทาง'] = np.where(df.contribution > 0, 'ดันไป spam', 'ดันไป ham')
            lim = max(1.0, float(df.contribution.abs().max()) * 1.15)
            chart = (
                alt.Chart(df).mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X('contribution:Q', title='← ham    |    spam →', scale=alt.Scale(domain=[-lim, lim]), axis=alt.Axis(grid=False)),
                    y=alt.Y('term:N', sort=alt.EncodingSortField(field='contribution', order='descending'), title=None, axis=alt.Axis(labelLimit=220)),
                    color=alt.Color('ทิศทาง:N', scale=alt.Scale(domain=['ดันไป spam', 'ดันไป ham'], range=['#EA580C', '#2563EB']), legend=None),
                    tooltip=['term', alt.Tooltip('contribution:Q', format='+.2f', title='ผลต่อคะแนน')],
                )
                .properties(height=max(120, 34 * len(df)))
            )
            try:
                st.altair_chart(chart, width='stretch')
            except TypeError:
                st.altair_chart(chart, use_container_width=True)
            st.markdown(f'<div class="note">คะแนนรวม = ผลรวมของทุกคำ ({score - intercept:+.2f}) + ค่าเริ่มต้นของโมเดล ({intercept:+.2f}) = <b>{score:+.2f}</b> '
                        '(คะแนนไม่ใช่ความน่าจะเป็น)</div>', unsafe_allow_html=True)
            st.markdown('<div class="note">ผลนี้เป็นการประมาณของโมเดล ไม่ได้พิสูจน์ว่าข้อความเป็นมิจฉาชีพหรือไม่ และ "คำที่มีผล" คือน้ำหนักของโมเดล ไม่ใช่เหตุผลจริงของข้อความ</div>',
                        unsafe_allow_html=True)

with tab_about:
    m = meta['test_metrics']['LogReg']
    st.markdown('<div class="section-label">โมเดลนี้เก่งแค่ไหน (ชุดทดสอบ 742 ข้อความ, spam 93)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f"{m['precision']:.3f}", help='ที่โมเดลว่า spam แล้ว spam จริงกี่ส่วน')
    c2.metric('Recall', f"{m['recall']:.3f}", help='spam ทั้งหมดที่มี จับได้กี่ส่วน')
    c3.metric('F1', f"{m['f1']:.3f}", help='ค่าเฉลี่ยฮาร์มอนิกที่ช่วยสมดุล precision และ recall')
    st.caption(f"ข้อความปกติที่ถูกจับผิดเป็น spam {m['FP']} ข้อความ · spam ที่หลุดไป {m['FN']} ข้อความ · "
               'เทียบกับการทาย ham ทุกข้อ: accuracy 87.5% แต่จับ spam ไม่ได้เลย')

    st.markdown('<div class="section-label">ทำงานยังไง</div>', unsafe_allow_html=True)
    st.markdown(
        '1. แปลงข้อความเป็นตัวเลขด้วย **TF-IDF** (นับคำเดี่ยวและคู่คำ)\n'
        '2. **Logistic Regression** ให้น้ำหนักแต่ละคำ แล้วรวมเป็น "คะแนน"\n'
        '3. คะแนนมากกว่า 0 = spam, น้อยกว่า 0 = ham\n\n'
        'โมเดลนี้เทรนเองจากข้อมูลจริง ไม่ได้เรียก LLM และโหลดโมเดลที่เทรนไว้ ไม่เทรนใหม่ตอนกดตรวจ')

    st.markdown('<div class="section-label">ข้อจำกัดที่ควรรู้</div>', unsafe_allow_html=True)
    st.markdown(
        '- รองรับ**ภาษาอังกฤษเท่านั้น**\n'
        '- ระบบแสดง **ประเมินไม่ได้** เมื่อเป็นภาษาอื่น ไม่พบคำที่รู้จัก หรือยาวเกิน 5,000 ตัวอักษร\n'
        '- ออกแบบสำหรับ **SMS สั้น** ข้อความยาวกว่า 1,000 ตัวอักษรจะมีคำเตือน\n'
        '- ข้อมูลเทรนเก่า (ราวปี 2000s) ไม่ได้แทน SMS สแปมปัจจุบัน\n'
        '- โมเดลไม่ได้ถูกทุกข้อความ และ**ไม่ได้พิสูจน์ว่าเป็นมิจฉาชีพ**\n'
        '- "คำที่มีผล" คือน้ำหนักของโมเดล ไม่ใช่เหตุผลจริงของข้อความนั้น\n'
        '- คะแนนไม่ใช่ความน่าจะเป็น')
    st.caption('ข้อมูล: Almeida, Gómez Hidalgo & Yamakami, UCI SMS Spam Collection (CC BY 4.0) · โปรเจกต์ CP020003')
