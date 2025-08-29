import os, io, re, json, csv, time, uuid, math, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, send_file, Response, redirect, url_for
from werkzeug.utils import secure_filename

import docx2txt, pdfplumber
from openai import OpenAI

# ===== 基础配置 =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_CHEAP  = os.getenv("LLM_MODEL_CHEAP", "deepseek-chat")
MAX_FILES        = int(os.getenv("MAX_FILES", "150"))
MAX_CONTENT_CHARS= int(os.getenv("MAX_CONTENT_CHARS", "40000"))
MUST_HAVE_CAP    = int(os.getenv("MUST_HAVE_CAP", "60"))
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", "3"))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET", "dev_secret")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=OPENAI_BASE_URL)

# ===== 进程内 batch 状态（简单内存存储，个人版够用）=====
BATCHES = {}  # {batch_id: {"jd":str,"notes":str,"must":list,"rows":[...], "done":bool, "csv":BytesIO}}

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".doc"}
def allowed(filename): return any(filename.lower().endswith(ext) for ext in ALLOWED_EXT)

def parse_text_from_file(fs):
    _, ext = os.path.splitext(secure_filename(fs.filename).lower())
    content = ""
    if ext == ".pdf":
        try:
            with pdfplumber.open(fs.stream) as pdf:
                pages = [(p.extract_text() or "") for p in pdf.pages]
            content = "\n".join(pages)
        except Exception: content = ""
    elif ext == ".docx":
        data = fs.read(); fs.stream.seek(0)
        try: content = docx2txt.process(io.BytesIO(data)) or ""
        except Exception: content = ""
    elif ext in [".txt", ".doc"]:
        try:
            content = fs.read().decode("utf-8", errors="ignore")
            fs.stream.seek(0)
        except Exception: content = ""
    content = re.sub(r"\s+", " ", (content or "")).strip()
    if len(content) > MAX_CONTENT_CHARS: content = content[:MAX_CONTENT_CHARS] + "\n[TRUNCATED]"
    return content

def extract_must_from_notes(notes: str):
    must = []
    for line in (notes or "").splitlines():
        s = line.strip()
        if not s: continue
        if re.match(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?', s, flags=re.I):
            s = re.sub(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?\s*', '', s, flags=re.I)
            if s: must.append(s)
    return must

# —— 年龄推算（按“本科/学士 + 年份区间/入学年”）——
UNDERGRAD_PAT = re.compile(r'(本科|学士|Bachelor)[^0-9]{0,12}(\d{4})(?:\D{0,3}(\d{4}))?', re.I)
def estimate_age_from_text(txt: str):
    m = UNDERGRAD_PAT.search(txt or "")
    if not m: return "不详"
    start = m.group(2); end = m.group(3)
    try:
        start_year = int(start)
        birth_year = start_year - 18
        return f"约{birth_year}年生"
    except Exception:
        return "不详"

def build_prompt(jd_text: str, notes: str, must_list: list[str], resume_text: str):
    sys_prompt = (
        "You are a professional recruiter across industries. "
        "Evaluate the resume against the JD and notes. "
        f"If any MUST-HAVE is missing, overall score must be <= {MUST_HAVE_CAP}. "
        "Follow the output schema strictly in JSON. "
        "Estimate age ONLY from undergraduate enrollment year (assume 18 at enrollment). "
        "If you cannot find undergrad enrollment, return '不详' for age."
    )
    user_prompt = f"""
[JD]
{jd_text or '(none)'}

[NOTES]
{notes or '(none)'}

[MUST-HAVE]
- {(chr(10)+'- ').join(must_list) if must_list else '(none)'}

[RESUME]
{resume_text}

[OUTPUT SCHEMA]
{{
  "name": "候选人姓名或未知",
  "education_brief": "1-2行教育背景要点（包含本科/研究生/博士及年份）",
  "estimated_age": "约1989年生 或 不详",
  "summary": "2-3行履历概要（公司/职能/年限/领域）",
  "highlights": ["亮点1","亮点2","亮点3"],
  "fit_analysis": "1-2段，结合JD说明匹配与不匹配点；若缺MUST，明确指出",
  "overall": 0-100,
  "evidence": ["“原文摘录A …”","“原文摘录B …”"],
  "risk_notes": ["稳定性/跳槽频率/时间线缺口等（如有）"]
}}
    """.strip()
    return sys_prompt, user_prompt

def call_llm(sys_prompt, user_prompt, retries=2):
    for i in range(retries+1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_CHEAP,
                messages=[{"role":"system","content":sys_prompt},
                          {"role":"user","content":user_prompt},
                          {"role":"user","content":"Return STRICT JSON only."}],
                temperature=0.2, max_tokens=1200
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}\s*$', raw, re.S)
            if m: raw = m.group(0)
            data = json.loads(raw)
            # 兜底字段
            data.setdefault("name","未知")
            data.setdefault("education_brief","")
            data.setdefault("estimated_age","不详")
            data.setdefault("summary","")
            data.setdefault("highlights",[])
            data.setdefault("fit_analysis","")
            data.setdefault("overall",0)
            data.setdefault("evidence",[])
            data.setdefault("risk_notes",[])
            return True, data
        except Exception as e:
            if i==retries: return False, {"error":f"LLM error: {e}"}
            time.sleep(0.8)

def process_one(file_storage, jd_text, notes, must_list):
    text = parse_text_from_file(file_storage)
    if not text:
        return {"filename": file_storage.filename, "ok": False, "data": {
            "name":"未知","education_brief":"","estimated_age":"不详","summary":"",
            "highlights":[],"fit_analysis":"","overall":0,"evidence":[],
            "risk_notes":[ "解析失败或空文档" ]
        }}
    # 如果模型没识别出年龄，我们再用正则兜底
    sys_p, user_p = build_prompt(jd_text, notes, must_list, text)
    ok, data = call_llm(sys_p, user_p)
    if ok and (not data.get("estimated_age") or data["estimated_age"]=="不详"):
        data["estimated_age"] = estimate_age_from_text(text)
    status = {"filename": file_storage.filename, "ok": ok, "data": data}
    if not ok:
        status["data"] = {"name":"未知","education_brief":"","estimated_age":"不详",
                          "summary":"","highlights":[],"fit_analysis":"",
                          "overall":0,"evidence":[], "risk_notes":[data.get("error","LLM失败")]}
    return status

# ========= 路由 =========
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 1) 提交后创建批次并跳转到结果页
@app.route("/start", methods=["POST"])
def start():
    jd_text = request.form.get("jd_raw","").strip()
    notes   = request.form.get("notes","").strip()
    files = [f for f in request.files.getlist("resumes") if f and allowed(f.filename)]
    if not files: return "No valid files. Allowed: .pdf, .docx, .txt, .doc", 400
    if len(files) > MAX_FILES: return f"Too many files (>{MAX_FILES}).", 400

    batch_id = str(uuid.uuid4())
    must_list = extract_must_from_notes(notes)
    BATCHES[batch_id] = {"jd": jd_text, "notes": notes, "must": must_list, "rows": [], "done": False, "csv": None}

    # 后台线程并发处理
    def run_batch():
        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(process_one, f, jd_text, notes, must_list) for f in files]
            for fu in as_completed(futs):
                r = fu.result()
                rows.append(r)
                BATCHES[batch_id]["rows"].append(r)   # 推进流
        # 排序 & 生成CSV
        rows.sort(key=lambda x: x["data"].get("overall",0), reverse=True)
        csv_buf = io.StringIO(); w = csv.writer(csv_buf)
        w.writerow(["filename","name","estimated_age","overall","education_brief","summary","highlights","fit_analysis","risk_notes"])
        for r in rows:
            d=r["data"]
            w.writerow([r["filename"], d.get("name",""), d.get("estimated_age",""), d.get("overall",0),
                        d.get("education_brief",""), d.get("summary",""),
                        " | ".join(d.get("highlights",[])),
                        d.get("fit_analysis","").replace("\n"," "),
                        " | ".join(d.get("risk_notes",[]))])
        BATCHES[batch_id]["csv"] = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
        BATCHES[batch_id]["done"] = True

    threading.Thread(target=run_batch, daemon=True).start()
    # 跳到结果页，前端用 SSE 订阅
    return redirect(url_for("results", batch_id=batch_id))

# 2) 结果页
@app.route("/results/<batch_id>")
def results(batch_id):
    b = BATCHES.get(batch_id)
    if not b: return "Batch not found", 404
    return render_template("results.html", batch_id=batch_id, jd=b["jd"], notes=b["notes"], must=b["must"])

# 3) SSE：流式推送每条结果
@app.route("/events/<batch_id>")
def events(batch_id):
    def gen():
        sent = 0
        while True:
            b = BATCHES.get(batch_id)
            if not b: break
            rows = b["rows"]
            # 推送新到达的
            while sent < len(rows):
                payload = rows[sent]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                sent += 1
            if b["done"]:
                yield "event: done\ndata: end\n\n"
                break
            time.sleep(0.8)
    return Response(gen(), mimetype="text/event-stream")

# 4) 下载CSV
@app.route("/download_csv/<batch_id>")
def download_csv(batch_id):
    b = BATCHES.get(batch_id)
    if not b or not b["csv"]: return "CSV not ready", 404
    b["csv"].seek(0)
    return send_file(b["csv"], as_attachment=True, download_name=f"results_{batch_id[:8]}.csv", mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
