import os, io, re, json, csv, time, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, send_file, Response, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

import docx2txt, pdfplumber
from openai import OpenAI
from openpyxl import Workbook  # Excel 导出

# ===================== 基础配置（来自环境变量） =====================
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_CHEAP    = os.getenv("LLM_MODEL_CHEAP", "deepseek-chat")
MAX_FILES          = int(os.getenv("MAX_FILES", "150"))
MAX_CONTENT_CHARS  = int(os.getenv("MAX_CONTENT_CHARS", "40000"))
MUST_HAVE_CAP      = int(os.getenv("MUST_HAVE_CAP", "60"))
MAX_WORKERS        = int(os.getenv("MAX_WORKERS", "3"))
STREAM_MODE        = os.getenv("STREAM_MODE", "sse")  # "sse" 或 "poll"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.secret_key = os.getenv("FLASK_SECRET", "dev_secret")

# OpenAI 兼容 DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=OPENAI_BASE_URL)

# 批次状态（内存版，个人使用足够）
BATCHES = {}  # {batch_id: {"jd":str,"notes":str,"must":list,"rows":[...], "done":bool, "csv":BytesIO, "excel":BytesIO}}

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".doc"}
def allowed(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXT)

# ============== 小工具：更鲁棒的 JSON 解析 ==============
def _safe_json_parse(raw: str):
    """
    依次尝试：
    1) 直接 json.loads
    2) 去除 ```(json) 围栏后再 loads
    3) 抓取第一个 {...} 块
    4) 截取从第一个 { 到最后一个 } 的内容
    """
    import json, re
    if raw is None:
        raise ValueError("empty content")
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    try:
        return json.loads(s2)
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, re.S)
    if m:
        return json.loads(m.group(0))

    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(s[i:j+1])

    # 仍失败：抛出 JSONDecodeError
    return json.loads(s)

# ===================== 解析：以“字节 blob”为输入（含 PDF 兜底） =====================
def parse_text_from_blob(filename: str, blob: bytes) -> str:
    """
    尽量从 PDF/DOCX/TXT 提取文本：
    - PDF: 先 pdfplumber（pdfminer），若为空或过短，再用 pypdfium2 兜底
    - DOCX: docx2txt
    - TXT/DOC: 尝试 utf-8 => latin-1
    """
    name = secure_filename(filename).lower()
    _, ext = os.path.splitext(name)
    content = ""

    try:
        if ext == ".pdf":
            # 1) pdfplumber 优先
            try:
                with pdfplumber.open(io.BytesIO(blob)) as pdf:
                    pages = [(p.extract_text() or "") for p in pdf.pages]
                content = "\n".join(pages).strip()
            except Exception:
                content = ""

            # 2) 兜底 pypdfium2（文字型 PDF 成功率更高；扫描件仍无解）
            if len(content) < 40:
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(io.BytesIO(blob))
                    texts = []
                    for i in range(len(pdf)):
                        page = pdf[i]
                        txtpage = page.get_textpage()
                        texts.append(txtpage.get_text_range())
                        txtpage.close()
                    content = "\n".join(texts).strip()
                except Exception:
                    pass

        elif ext == ".docx":
            try:
                content = docx2txt.process(io.BytesIO(blob)) or ""
            except Exception:
                content = ""

        elif ext in [".txt", ".doc"]:
            try:
                content = blob.decode("utf-8", errors="ignore")
            except Exception:
                content = blob.decode("latin-1", errors="ignore")
        else:
            content = ""

    except Exception:
        content = ""

    # 清洗 & 截断
    content = re.sub(r"\s+", " ", (content or "")).strip()
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n[TRUNCATED]"
    return content

# ===================== MUST 规则抽取 =====================
def extract_must_from_notes(notes: str):
    must = []
    for line in (notes or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?', s, flags=re.I):
            s = re.sub(r'^(!|\[?MUST\]?|必须|必需|必备)\s*[:：\-]?\s*', '', s, flags=re.I)
            if s:
                must.append(s)
    return must

# 本科入学年→年龄估算
UNDERGRAD_PAT = re.compile(r'(本科|学士|Bachelor)[^0-9]{0,12}(\d{4})(?:\D{0,3}(\d{4}))?', re.I)
def estimate_age_from_text(txt: str):
    m = UNDERGRAD_PAT.search(txt or "")
    if not m:
        return "不详"
    try:
        start_year = int(m.group(2))
        birth_year = start_year - 18
        return f"约{birth_year}年生"
    except Exception:
        return "不详"

# ===================== 提示词与 LLM 调用 =====================
def build_prompt(jd_text: str, notes: str, must_list: list[str], resume_text: str):
    sys_prompt = (
        "You are a professional recruiter across industries. "
        "Evaluate the resume against the JD and notes. "
        f"If any MUST-HAVE is missing, overall score must be <= {MUST_HAVE_CAP}. "
        "Derive criteria from the JD; do not use ATS rubrics. "
        "Return STRICT JSON only. "
        "Estimate age ONLY from undergraduate enrollment year (assume 18). If unknown, return '不详'."
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

def call_llm(sys_prompt: str, user_prompt: str, retries: int = 3):
    """
    更稳健的 LLM 调用：
    - 强制 JSON 输出（response_format）
    - 失败退避重试
    - 解析时做多重兜底
    """
    backoff = 0.8
    last_err = None
    raw_preview = ""
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_CHEAP,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},  # 强制 JSON
            )
            raw = resp.choices[0].message.content
            raw_preview = (raw or "")[:200]
            data = _safe_json_parse(raw or "")
            # 字段兜底
            data.setdefault("name", "未知")
            data.setdefault("education_brief", "")
            data.setdefault("estimated_age", "不详")
            data.setdefault("summary", "")
            data.setdefault("highlights", [])
            data.setdefault("fit_analysis", "")
            data.setdefault("overall", 0)
            data.setdefault("evidence", [])
            data.setdefault("risk_notes", [])
            return True, data
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 1.6
    # 全部失败，回传错误 + 原始预览
    return False, {"error": f"{type(last_err).__name__}: {last_err} | raw: {raw_preview}"}

def process_one(blob_item: dict, jd_text: str, notes: str, must_list: list[str]):
    filename = blob_item["filename"]
    blob = blob_item["blob"]

    text = parse_text_from_blob(filename, blob)
    if not text:
        return {
            "filename": filename,
            "ok": False,
            "data": {
                "name": "未知",
                "education_brief": "",
                "estimated_age": "不详",
                "summary": "",
                "highlights": [],
                "fit_analysis": "",
                "overall": 0,
                "evidence": [],
                "risk_notes": ["解析失败或空文档"],
            },
        }

    sys_p, user_p = build_prompt(jd_text, notes, must_list, text)
    ok, data = call_llm(sys_p, user_p)

    if ok and (not data.get("estimated_age") or data["estimated_age"] == "不详"):
        # 正则兜底年龄估计
        data["estimated_age"] = estimate_age_from_text(text)

    if not ok:
        err = data.get("error", "LLM失败")
        return {
            "filename": filename,
            "ok": False,
            "data": {
                "name": "未知",
                "education_brief": "",
                "estimated_age": "不详",
                "summary": "",
                "highlights": [],
                "fit_analysis": "",
                "overall": 0,
                "evidence": [],
                "risk_notes": [err],
            },
        }

    return {"filename": filename, "ok": True, "data": data}

# ===================== 路由 =====================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 提交后创建批次（把上传文件读到内存 blobs），后台并发处理
@app.route("/start", methods=["POST"])
def start():
    jd_text = request.form.get("jd_raw", "").strip()
    notes = request.form.get("notes", "").strip()

    uploads = []
    for f in request.files.getlist("resumes"):
        if not f or not allowed(f.filename):
            continue
        b = f.read()  # 读成字节，避免后续句柄被关闭
        uploads.append({"filename": f.filename, "blob": b})
    if not uploads:
        return "No valid files. Allowed: .pdf, .docx, .txt, .doc", 400
    if len(uploads) > MAX_FILES:
        return f"Too many files (>{MAX_FILES}).", 400

    batch_id = str(uuid.uuid4())
    must_list = extract_must_from_notes(notes)
    BATCHES[batch_id] = {
        "jd": jd_text,
        "notes": notes,
        "must": must_list,
        "rows": [],
        "done": False,
        "csv": None,
        "excel": None,
    }

    def run_batch():
        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(process_one, item, jd_text, notes, must_list) for item in uploads]
            for fu in as_completed(futs):
                r = fu.result()
                rows.append(r)
                # 供流式/轮询实时读取
                BATCHES[batch_id]["rows"].append(r)

        rows.sort(key=lambda x: x["data"].get("overall", 0), reverse=True)

        # 生成 CSV
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        w.writerow(
            [
                "filename",
                "name",
                "estimated_age",
                "overall",
                "education_brief",
                "summary",
                "highlights",
                "fit_analysis",
                "risk_notes",
            ]
        )
        for r in rows:
            d = r["data"]
            w.writerow(
                [
                    r["filename"],
                    d.get("name", ""),
                    d.get("estimated_age", ""),
                    d.get("overall", 0),
                    d.get("education_brief", ""),
                    d.get("summary", ""),
                    " | ".join(d.get("highlights", [])),
                    (d.get("fit_analysis", "") or "").replace("\n", " "),
                    " | ".join(d.get("risk_notes", [])),
                ]
            )
        BATCHES[batch_id]["csv"] = io.BytesIO(csv_buf.getvalue().encode("utf-8"))

        # 生成 Excel（推荐导出）
        wb = Workbook(); ws = wb.active
        ws.append(["文件名","姓名","年龄","教育背景","履历概要","亮点","匹配度","匹配度分析","证据","风险备注"])
        for r in rows:
            d = r["data"]
            ws.append([
                r["filename"],
                d.get("name",""),
                d.get("estimated_age",""),
                d.get("education_brief",""),
                d.get("summary",""),
                " | ".join(d.get("highlights",[])),
                d.get("overall",0),
                d.get("fit_analysis",""),
                " | ".join(d.get("evidence",[])),
                " | ".join(d.get("risk_notes",[])),
            ])
        excel_buf = io.BytesIO(); wb.save(excel_buf); excel_buf.seek(0)
        BATCHES[batch_id]["excel"] = excel_buf

        BATCHES[batch_id]["done"] = True

    threading.Thread(target=run_batch, daemon=True).start()
    return redirect(url_for("results", batch_id=batch_id))

# 结果页（模板里会根据 stream_mode 决定 SSE 还是轮询）
@app.route("/results/<batch_id>")
def results(batch_id):
    b = BATCHES.get(batch_id)
    if not b:
        return "Batch not found", 404
    return render_template(
        "results.html",
        batch_id=batch_id,
        jd=b["jd"],
        notes=b["notes"],
        must=b["must"],
        stream_mode=STREAM_MODE,
    )

# SSE 流式事件
@app.route("/events/<batch_id>")
def events(batch_id):
    def gen():
        sent = 0
        while True:
            b = BATCHES.get(batch_id)
            if not b:
                break
            rows = b["rows"]
            while sent < len(rows):
                yield f"data: {json.dumps(rows[sent], ensure_ascii=False)}\n\n"
                sent += 1
            if b["done"]:
                yield "event: done\ndata: end\n\n"
                break
            time.sleep(0.8)
    return Response(gen(), mimetype="text/event-stream")

# 轮询：返回当前行与完成状态
@app.route("/rows/<batch_id>")
def rows_json(batch_id):
    b = BATCHES.get(batch_id)
    if not b:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify({"rows": b["rows"], "done": b["done"]})

# 下载 CSV
@app.route("/download_csv/<batch_id>")
def download_csv(batch_id):
    b = BATCHES.get(batch_id)
    if not b or not b["csv"]:
        return "CSV not ready", 404
    b["csv"].seek(0)
    return send_file(
        b["csv"],
        as_attachment=True,
        download_name=f"results_{batch_id[:8]}.csv",
        mimetype="text/csv",
    )

# 下载 Excel
@app.route("/download_excel/<batch_id>")
def download_excel(batch_id):
    b = BATCHES.get(batch_id)
    if not b or not b.get("excel"):
        return "Excel not ready", 404
    b["excel"].seek(0)
    return send_file(
        b["excel"],
        as_attachment=True,
        download_name=f"results_{batch_id[:8]}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# 诊断：检查 Key/Endpoint/模型是否可用
@app.route("/diag")
def diag():
    info = {
        "has_key": bool(DEEPSEEK_API_KEY),
        "base_url": OPENAI_BASE_URL,
        "model": LLM_MODEL_CHEAP,
        "max_workers": MAX_WORKERS,
    }
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL_CHEAP, messages=[{"role": "user", "content": "ping"}], max_tokens=5,
            response_format={"type": "json_object"},
        )
        info["test"] = "ok"
    except Exception as e:
        info["test"] = f"error: {e}"
    return jsonify(info), 200


if __name__ == "__main__":
    # 本地调试
    app.run(host="0.0.0.0", port=5000, debug=True)
