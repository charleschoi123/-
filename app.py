import os, io, re, json, csv, time, uuid
from datetime import datetime
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename

# -------- 模型&解析依赖 --------
import docx2txt
import pdfplumber

# OpenAI SDK 兼容 DeepSeek（OpenAI风格）
from openai import OpenAI

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# DeepSeek 的 OpenAI 兼容 Endpoint（默认）
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_CHEAP  = os.getenv("LLM_MODEL_CHEAP", "deepseek-chat")   # 便宜/常用
MAX_FILES        = int(os.getenv("MAX_FILES", "150"))              # 单批最多文件
MAX_CONTENT_CHARS= int(os.getenv("MAX_CONTENT_CHARS", "40000"))    # 每份简历最大解析字符

# 评分权重（可在 .env 或 Render 环境变量里覆盖）
WEIGHTS_DEFAULT = {
    "skills": 0.40,
    "domain": 0.20,
    "stage": 0.15,
    "language": 0.10,
    "leadership": 0.10,
    "compliance": 0.05
}

# 必须项缺失时的上限分
MUST_HAVE_CAP = int(os.getenv("MUST_HAVE_CAP", "60"))

# ========== Flask ==========
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 批量上传
app.secret_key = os.getenv("FLASK_SECRET", "dev_secret")

# ========== OpenAI Client (DeepSeek) ==========
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=OPENAI_BASE_URL
)

# ========== 工具函数 ==========
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".doc"}  # .doc 将用纯文本兜底
def allowed(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXT)

def parse_text_from_file(file_storage):
    """将 PDF/DOCX/TXT 解析为纯文本；扫描版 PDF 暂不OCR（后续可加）"""
    filename = secure_filename(file_storage.filename)
    _, ext = os.path.splitext(filename.lower())
    content = ""

    if ext == ".pdf":
        try:
            with pdfplumber.open(file_storage.stream) as pdf:
                pages = []
                for p in pdf.pages:
                    pages.append(p.extract_text() or "")
                content = "\n".join(pages)
        except Exception as e:
            content = ""
    elif ext == ".docx":
        # 读取到 BytesIO 再给 docx2txt
        data = file_storage.read()
        file_storage.stream.seek(0)
        content = docx2txt.process(io.BytesIO(data)) or ""
    elif ext in [".txt", ".doc"]:
        # 简单读取（.doc 当 .txt 兜底）
        try:
            raw = file_storage.read().decode('utf-8', errors='ignore')
            file_storage.stream.seek(0)
            content = raw
        except Exception:
            content = ""
    else:
        content = ""

    # 清洗：去页眉页脚常见噪声，合并多空格
    content = re.sub(r'\s+', ' ', content).strip()
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n[TRUNCATED]"
    return content

def build_prompt(jd_struct, resume_text):
    """构造对 LLM 的提示：JD要点 + 简历全文（或截断）"""
    must_list = "\n".join(f"- {x}" for x in jd_struct.get("must_have", []))
    nice_list = "\n".join(f"- {x}" for x in jd_struct.get("nice_have", []))
    skills    = ", ".join(jd_struct.get("skills", []))
    domain    = jd_struct.get("domain", "")
    stage     = jd_struct.get("stage", "")
    lang_loc  = jd_struct.get("lang_loc", "")

    weights_json = json.dumps(jd_struct.get("weights", WEIGHTS_DEFAULT), ensure_ascii=False)

    sys_prompt = (
        "You are a hiring evaluator in biopharma. "
        "Score the candidate strictly with evidence from the resume text. "
        "If any MUST-HAVE is missing, overall score must be <= {cap}. "
        "Return STRICT JSON only; no extra text."
    ).format(cap=MUST_HAVE_CAP)

    user_prompt = f"""
[JD KEYPOINTS]
- Must-have:
{must_list if must_list else '(none)'}
- Nice-to-have:
{nice_list if nice_list else '(none)'}
- Skills: {skills or '(none)'}
- Domain/Therapy: {domain or '(none)'}
- Stage: {stage or '(none)'}
- Language/Location: {lang_loc or '(none)'}

[WEIGHTS]
{weights_json}

[RESUME FULLTEXT]
{resume_text}

[OUTPUT SCHEMA]
{{
 "overall": 0-100,
 "dimensions": {{
    "skills":0-100,"domain":0-100,"stage":0-100,"language":0-100,"leadership":0-100,"compliance":0-100
 }},
 "pros": ["..."],
 "gaps": ["..."],
 "evidence": ["公司/年份/职责要点：原文摘录..."],
 "next_actions": ["..."],
 "risk_notes": ["..."]
}}
    """.strip()

    return sys_prompt, user_prompt

def call_llm_for_json(sys_prompt, user_prompt, max_retries=2):
    """调用 DeepSeek（OpenAI风格），强制只返回JSON；失败重试并做简单JSON修复"""
    for i in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_CHEAP,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "user", "content": "Return STRICT JSON only without any extra commentary."}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            raw = resp.choices[0].message.content.strip()
            # 尝试截出最外层JSON
            m = re.search(r'\{.*\}\s*$', raw, re.S)
            if m:
                raw = m.group(0)
            data = json.loads(raw)
            return True, data, raw
        except Exception as e:
            if i == max_retries:
                return False, None, f"LLM error: {e}"
            time.sleep(0.8)
    return False, None, "Unknown error"

def normalize_result(js):
    """兜底：缺字段时补空，防止模板渲染报错"""
    dims_key = ["skills","domain","stage","language","leadership","compliance"]
    out = {
        "overall": js.get("overall", 0),
        "dimensions": {k: js.get("dimensions", {}).get(k, 0) for k in dims_key},
        "pros": js.get("pros", []),
        "gaps": js.get("gaps", []),
        "evidence": js.get("evidence", []),
        "next_actions": js.get("next_actions", []),
        "risk_notes": js.get("risk_notes", [])
    }
    return out

# ========== 路由 ==========
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", default_weights=json.dumps(WEIGHTS_DEFAULT, ensure_ascii=False))

@app.route("/analyze", methods=["POST"])
def analyze():
    # 1) 取 JD & Must/Nice 列表
    jd_raw = request.form.get("jd_raw", "").strip()
    must_have = [s.strip() for s in request.form.get("must_have","").split("\n") if s.strip()]
    nice_have = [s.strip() for s in request.form.get("nice_have","").split("\n") if s.strip()]
    skills    = [s.strip() for s in request.form.get("skills","").split(",") if s.strip()]
    domain    = request.form.get("domain","").strip()
    stage     = request.form.get("stage","").strip()
    lang_loc  = request.form.get("lang_loc","").strip()
    try:
        weights  = json.loads(request.form.get("weights_json","")) if request.form.get("weights_json","") else WEIGHTS_DEFAULT
    except:
        weights = WEIGHTS_DEFAULT

    jd_struct = {
        "jd_raw": jd_raw,
        "must_have": must_have,
        "nice_have": nice_have,
        "skills": skills,
        "domain": domain,
        "stage": stage,
        "lang_loc": lang_loc,
        "weights": weights
    }

    # 2) 文件批量
    files = request.files.getlist("resumes")
    files = [f for f in files if f and allowed(f.filename)]
    if not files:
        return "No valid files. Allowed: .pdf, .docx, .txt, .doc", 400
    if len(files) > MAX_FILES:
        return f"Too many files (>{MAX_FILES}).", 400

    # 3) 逐份解析 & 打分
    results = []
    for f in files:
        text = parse_text_from_file(f)
        if not text:
            results.append({
                "filename": f.filename,
                "ok": False,
                "error": "解析失败或为空",
                "raw_json": "{}",
                "normalized": normalize_result({})
            })
            continue

        sys_p, user_p = build_prompt(jd_struct, text)
        ok, data, raw = call_llm_for_json(sys_p, user_p)

        if not ok:
            results.append({
                "filename": f.filename,
                "ok": False,
                "error": raw,
                "raw_json": "{}",
                "normalized": normalize_result({})
            })
        else:
            results.append({
                "filename": f.filename,
                "ok": True,
                "error": "",
                "raw_json": json.dumps(data, ensure_ascii=False, indent=2),
                "normalized": normalize_result(data)
            })

    # 4) 排序（overall desc）
    results.sort(key=lambda r: r["normalized"]["overall"], reverse=True)

    # 5) CSV 导出缓存（内存生成）
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow([
        "filename","overall","skills","domain","stage","language","leadership","compliance",
        "pros","gaps","next_actions","risk_notes"
    ])
    for r in results:
        n = r["normalized"]
        dims = n["dimensions"]
        writer.writerow([
            r["filename"], n["overall"], dims["skills"], dims["domain"], dims["stage"],
            dims["language"], dims["leadership"], dims["compliance"],
            " | ".join(n["pros"]), " | ".join(n["gaps"]),
            " | ".join(n["next_actions"]), " | ".join(n["risk_notes"])
        ])
    csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
    csv_id = str(uuid.uuid4())
    app.config[f"CSV_{csv_id}"] = csv_bytes  # 简易缓存（Render 单实例够用）

    return render_template("results.html",
                           jd=jd_struct,
                           results=results,
                           csv_id=csv_id,
                           ts=datetime.utcnow().isoformat()+"Z")

@app.route("/download_csv/<csv_id>")
def download_csv(csv_id):
    buf = app.config.get(f"CSV_{csv_id}")
    if not buf:
        return "CSV expired.", 404
    buf.seek(0)
    filename = f"results_{csv_id[:8]}.csv"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="text/csv")


if __name__ == "__main__":
    # 本地调试
    app.run(host="0.0.0.0", port=5000, debug=True)
