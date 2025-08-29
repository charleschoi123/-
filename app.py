import os, io, re, json, csv, time, uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

import docx2txt
import pdfplumber
from openai import OpenAI

# ========= 配置 =========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_CHEAP  = os.getenv("LLM_MODEL_CHEAP", "deepseek-chat")
MAX_FILES        = int(os.getenv("MAX_FILES", "150"))
MAX_CONTENT_CHARS= int(os.getenv("MAX_CONTENT_CHARS", "40000"))
MUST_HAVE_CAP    = int(os.getenv("MUST_HAVE_CAP", "60"))
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", "3"))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.secret_key = os.getenv("FLASK_SECRET", "dev_secret")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=OPENAI_BASE_URL)

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".doc"}
def allowed(filename): return any(filename.lower().endswith(ext) for ext in ALLOWED_EXT)

def parse_text_from_file(file_storage):
    filename = secure_filename(file_storage.filename)
    _, ext = os.path.splitext(filename.lower())
    content = ""
    if ext == ".pdf":
        try:
            with pdfplumber.open(file_storage.stream) as pdf:
                pages = [(p.extract_text() or "") for p in pdf.pages]
            content = "\n".join(pages)
        except Exception:
            content = ""
    elif ext == ".docx":
        data = file_storage.read(); file_storage.stream.seek(0)
        try: content = docx2txt.process(io.BytesIO(data)) or ""
        except Exception: content = ""
    elif ext in [".txt", ".doc"]:
        try:
            raw = file_storage.read().decode('utf-8', errors='ignore')
            file_storage.stream.seek(0); content = raw
        except Exception:
            content = ""
    content = re.sub(r'\s+', ' ', (content or "")).strip()
    if len(content) > MAX_CONTENT_CHARS: content = content[:MAX_CONTENT_CHARS] + "\n[TRUNCATED]"
    return content

def extract_must_from_notes(notes: str):
    """从补充说明里抽取必须项：支持前缀 '必须' '必需' '必备' '!' '[MUST]' 等"""
    must = []
    for line in (notes or "").splitlines():
        s = line.strip()
        if not s: continue
        if re.match(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?', s, flags=re.I):
            s = re.sub(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?\s*', '', s, flags=re.I)
            if s: must.append(s)
    return must

def build_prompt(jd_text: str, notes: str, inferred_must: list[str]):
    sys_prompt = (
        "You are a professional recruiter for multi-industry roles. "
        "Given a job description (JD) and optional notes, evaluate a resume and return ONLY JSON. "
        f"If any MUST-HAVE items are missing, the overall score must be <= {MUST_HAVE_CAP}. "
        "Derive criteria from the JD itself (no ATS rubric). Provide brief evidence quotes from the resume."
    )

    user_prompt = f"""
[JD]
{jd_text or '(none)'}

[NOTES]
{notes or '(none)'}

[MUST-HAVE extracted from NOTES]
- {(chr(10)+'- ').join(inferred_must) if inferred_must else '(none)'}

[RESUME]
{{RESUME_TEXT}}

[OUTPUT SCHEMA]
{{
 "overall": 0-100,
 "pros": ["..."],
 "gaps": ["..."],
 "evidence": ["原文摘录..."],
 "next_actions": ["..."],
 "risk_notes": ["..."]
}}
""".strip()
    return sys_prompt, user_prompt

def call_llm_for_json(sys_prompt, user_prompt_filled, max_retries=2):
    for i in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_CHEAP,
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_prompt_filled},
                    {"role":"user","content":"Return STRICT JSON only without any extra commentary."}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}\s*$', raw, re.S)
            if m: raw = m.group(0)
            data = json.loads(raw)
            # 兜底字段
            out = {
                "overall": data.get("overall", 0),
                "pros": data.get("pros", []),
                "gaps": data.get("gaps", []),
                "evidence": data.get("evidence", []),
                "next_actions": data.get("next_actions", []),
                "risk_notes": data.get("risk_notes", []),
            }
            return True, out, json.dumps(out, ensure_ascii=False, indent=2)
        except Exception as e:
            if i == max_retries: return False, None, f"LLM error: {e}"
            time.sleep(0.8)
    return False, None, "Unknown error"

def process_one(file_storage, jd_text, notes, inferred_must):
    text = parse_text_from_file(file_storage)
    if not text:
        return {
            "filename": file_storage.filename,
            "ok": False,
            "error": "解析失败或为空",
            "raw_json": "{}",
            "normalized": {"overall":0,"pros":[],"gaps":[],"evidence":[],"next_actions":[],"risk_notes":[]}
        }
    sys_p, user_p_tpl = build_prompt(jd_text, notes, inferred_must)
    user_filled = user_p_tpl.replace("{RESUME_TEXT}", text)
    ok, data, raw = call_llm_for_json(sys_p, user_filled)
    if not ok:
        return {"filename": file_storage.filename, "ok": False, "error": raw, "raw_json": "{}", "normalized": {"overall":0,"pros":[],"gaps":[],"evidence":[],"next_actions":[],"risk_notes":[]}}
    else:
        return {"filename": file_storage.filename, "ok": True, "error": "", "raw_json": raw, "normalized": data}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    jd_text = request.form.get("jd_raw","").strip()
    notes   = request.form.get("notes","").strip()  # 仅保留 JD + 补充说明
    files = [f for f in request.files.getlist("resumes") if f and allowed(f.filename)]
    if not files: return "No valid files. Allowed: .pdf, .docx, .txt, .doc", 400
    if len(files) > MAX_FILES: return f"Too many files (>{MAX_FILES}).", 400

    inferred_must = extract_must_from_notes(notes)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one, f, jd_text, notes, inferred_must) for f in files]
        for fu in as_completed(futs):
            results.append(fu.result())

    results.sort(key=lambda r: r["normalized"]["overall"], reverse=True)

    # CSV
    csv_buf = io.StringIO(); writer = csv.writer(csv_buf)
    writer.writerow(["filename","overall","pros","gaps","next_actions","risk_notes"])
    for r in results:
        n = r["normalized"]
        writer.writerow([
            r["filename"], n["overall"],
            " | ".join(n["pros"]), " | ".join(n["gaps"]),
            " | ".join(n["next_actions"]), " | ".join(n["risk_notes"])
        ])
    csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
    csv_id = str(uuid.uuid4()); app.config[f"CSV_{csv_id}"] = csv_bytes

    return render_template("results.html",
                           jd={"jd_raw": jd_text, "notes": notes, "must": inferred_must},
                           results=results, csv_id=csv_id,
                           ts=datetime.utcnow().isoformat()+"Z")

@app.route("/download_csv/<csv_id>")
def download_csv(csv_id):
    buf = app.config.get(f"CSV_{csv_id}")
    if not buf: return "CSV expired.", 404
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"results_{csv_id[:8]}.csv", mimetype="text/csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
