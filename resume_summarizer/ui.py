from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory
from .core import ResumeSummaryError, discover_pdfs, run_batch


DOWNLOAD_NAMES = {
    "resume_summary.pdf",
    "resume_summary.html",
    "resume_summary.csv",
    "results.json",
}


def _output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(
        os.environ.get("RESUME_SUMMARY_OUTPUT_ROOT", str(Path.home() / "Downloads"))
    ).expanduser()
    base = output_root / f"Resume_Summary_{stamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _unique_upload_name(upload_dir: Path, original_name: str) -> Path:
    basename = Path(original_name.replace("\\", "/")).name
    safe = unicodedata.normalize("NFKC", basename)
    safe = re.sub(r"[^\w .()（）\-]+", "_", safe, flags=re.UNICODE).strip(" .")
    safe = safe[-180:] or "resume.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    candidate = upload_dir / safe
    suffix = 2
    while candidate.exists():
        candidate = upload_dir / f"{Path(safe).stem}-{suffix}.pdf"
        suffix += 1
    return candidate


def create_app(prefill_path: str = "") -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024
    app.config["RESUME_TOKEN"] = secrets.token_urlsafe(24)
    app.config["PREFILL_PATH"] = prefill_path
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def authorized() -> bool:
        supplied = request.headers.get("X-Resume-Token") or request.args.get("token", "")
        return secrets.compare_digest(str(supplied), app.config["RESUME_TOKEN"])

    def update_job(job_id: str, **changes: Any) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update(changes)

    def process_job(
        job_id: str,
        pdfs: list[Path],
        output_dir: Path,
        upload_dir: Path | None,
        use_cache: bool,
    ) -> None:
        def on_progress(event: dict[str, Any]) -> None:
            event_name = event.get("event")
            if event_name == "models_ready":
                update_job(
                    job_id,
                    phase="模型已就绪",
                    models=event.get("models", []),
                )
            elif event_name == "file_start":
                total = max(int(event.get("total", len(pdfs))), 1)
                index = int(event.get("index", 1))
                update_job(
                    job_id,
                    phase="正在提取",
                    current_file=event.get("file", ""),
                    completed=index - 1,
                    total=total,
                    progress=round((index - 1) / total * 100),
                )
            elif event_name == "file_done":
                total = max(int(event.get("total", len(pdfs))), 1)
                index = int(event.get("index", 0))
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id]["records"] = [
                            *jobs[job_id].get("records", []),
                            event.get("record", {}),
                        ]
                        jobs[job_id].update(
                            completed=index,
                            total=total,
                            progress=round(index / total * 100),
                        )

        try:
            records = run_batch(
                pdfs,
                output_dir=output_dir,
                model="auto-free",
                provider="openrouter",
                progress=False,
                use_cache=use_cache,
                progress_callback=on_progress,
            )
            update_job(
                job_id,
                status="done",
                phase="处理完成",
                progress=100,
                completed=len(records),
                records=records,
                output_dir=str(output_dir),
                downloads=sorted(DOWNLOAD_NAMES),
            )
        except Exception as exc:
            message = str(exc) if isinstance(exc, ResumeSummaryError) else f"处理失败：{exc}"
            update_job(job_id, status="error", phase="处理失败", error=message)
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)

    @app.get("/")
    def home() -> str:
        token = app.config["RESUME_TOKEN"]
        prefill = app.config["PREFILL_PATH"]
        return _page_html(token=token, prefill_path=prefill)

    @app.post("/api/jobs")
    def create_job():
        if not authorized():
            abort(403)

        upload_dir: Path | None = None
        uploaded_paths: list[Path] = []
        files = [item for item in request.files.getlist("files") if item.filename]
        if files:
            upload_dir = Path(tempfile.mkdtemp(prefix="resume-summary-upload-"))
            for item in files:
                if Path(item.filename).suffix.lower() != ".pdf":
                    continue
                destination = _unique_upload_name(upload_dir, item.filename)
                item.save(destination)
                uploaded_paths.append(destination)

        path_text = request.form.get("folder_path", "").strip()
        local_paths: list[Path] = []
        if path_text:
            source = Path(path_text).expanduser()
            if not source.exists():
                if upload_dir:
                    shutil.rmtree(upload_dir, ignore_errors=True)
                return jsonify({"error": f"路径不存在：{path_text}"}), 400
            local_paths = discover_pdfs([source], recursive=source.is_dir())

        pdfs = sorted(
            {path.resolve(): path for path in [*local_paths, *uploaded_paths]}.values(),
            key=lambda path: path.name.casefold(),
        )
        if not pdfs:
            if upload_dir:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return jsonify({"error": "没有找到 PDF。请拖入文件、选择文件夹，或填写本机路径。"}), 400

        job_id = uuid.uuid4().hex
        output_dir = _output_dir()
        with jobs_lock:
            jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "phase": "正在检查当前免费模型",
                "progress": 0,
                "completed": 0,
                "total": len(pdfs),
                "current_file": "",
                "records": [],
                "models": [],
                "output_dir": str(output_dir),
                "downloads": [],
                "error": "",
            }
        worker = threading.Thread(
            target=process_job,
            args=(
                job_id,
                pdfs,
                output_dir,
                upload_dir,
                request.form.get("no_cache") != "1",
            ),
            daemon=True,
        )
        worker.start()
        return jsonify({"job_id": job_id})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        if not authorized():
            abort(403)
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404)
            return jsonify(dict(job))

    @app.get("/downloads/<job_id>/<name>")
    def download(job_id: str, name: str):
        if not authorized() or name not in DOWNLOAD_NAMES:
            abort(403)
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None or job.get("status") != "done":
                abort(404)
            output_dir = job["output_dir"]
        return send_from_directory(output_dir, name, as_attachment=True)

    @app.post("/api/jobs/<job_id>/reveal")
    def reveal(job_id: str):
        if not authorized():
            abort(403)
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None or job.get("status") != "done":
                abort(404)
            output_dir = job["output_dir"]
        if sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        return jsonify({"ok": True})

    @app.post("/api/shutdown")
    def shutdown():
        if not authorized():
            abort(403)
        threading.Timer(0.35, lambda: os._exit(0)).start()
        return jsonify({"ok": True})

    return app


def _page_html(token: str, prefill_path: str) -> str:
    import html
    import json

    safe_token = json.dumps(token)
    safe_prefill = html.escape(prefill_path, quote=True)
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resume Summary</title>
  <style>
    :root {{ color-scheme: light; --ink:#132238; --muted:#64748b; --line:#dce5ed; --navy:#153b5b; --blue:#246b91; --sky:#edf7fb; --ok:#167c5b; --bad:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink); background:linear-gradient(135deg,#e8f5f7 0%,#f8fbfd 45%,#edf0fa 100%); }}
    main {{ width:min(1080px,calc(100% - 32px)); margin:36px auto 64px; }}
    header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-bottom:22px; }}
    h1 {{ margin:0; font-size:clamp(29px,4vw,44px); line-height:1.08; letter-spacing:-.035em; color:var(--navy); }}
    .subtitle {{ margin:8px 0 0; color:var(--muted); }}
    .privacy {{ font-size:12px; color:var(--muted); background:#ffffffaa; border:1px solid #fff; padding:8px 12px; border-radius:999px; white-space:nowrap; }}
    .header-actions {{ display:flex; align-items:center; gap:8px; }}
    .quit {{ font-size:12px; padding:8px 11px; color:#7a2730; background:#fff8f8; }}
    .card {{ background:#fffffff2; border:1px solid #fff; box-shadow:0 18px 55px rgba(28,64,88,.12); border-radius:22px; padding:24px; backdrop-filter:blur(12px); }}
    #dropzone {{ border:2px dashed #8fb8c9; border-radius:17px; padding:38px 20px; text-align:center; background:var(--sky); transition:.18s ease; cursor:pointer; }}
    #dropzone.drag {{ transform:translateY(-2px); border-color:var(--blue); background:#e1f3fa; box-shadow:0 10px 26px rgba(36,107,145,.12); }}
    .drop-title {{ font-weight:720; font-size:19px; color:var(--navy); }}
    .drop-help {{ margin-top:7px; color:var(--muted); font-size:14px; }}
    .actions {{ display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:17px; }}
    button,.button {{ appearance:none; border:0; border-radius:10px; padding:10px 15px; font:inherit; font-weight:650; cursor:pointer; background:#fff; color:var(--navy); box-shadow:inset 0 0 0 1px var(--line); text-decoration:none; }}
    button:hover,.button:hover {{ filter:brightness(.98); transform:translateY(-1px); }}
    button.primary {{ color:#fff; background:var(--navy); box-shadow:none; padding:12px 20px; }}
    button:disabled {{ opacity:.52; cursor:not-allowed; transform:none; }}
    input[type=file] {{ display:none; }}
    .path-row {{ display:grid; grid-template-columns:1fr auto; gap:10px; margin-top:16px; }}
    input[type=text] {{ width:100%; border:1px solid var(--line); border-radius:11px; padding:12px 13px; background:#fff; color:var(--ink); font:inherit; outline:none; }}
    input[type=text]:focus {{ border-color:#6fa4bb; box-shadow:0 0 0 3px #d9eef6; }}
    .selection {{ min-height:22px; margin:12px 2px 0; color:var(--muted); font-size:13px; }}
    .controls {{ display:flex; justify-content:space-between; align-items:center; gap:14px; margin-top:17px; }}
    label.check {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px; }}
    .status {{ display:none; margin-top:20px; border-top:1px solid var(--line); padding-top:20px; }}
    .status.show {{ display:block; }}
    .status-head {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:9px; }}
    .phase {{ font-weight:700; }} .detail {{ color:var(--muted); font-size:13px; text-align:right; }}
    .bar {{ height:10px; background:#e8eef3; border-radius:999px; overflow:hidden; }}
    .bar > i {{ display:block; width:0; height:100%; background:linear-gradient(90deg,#1a7892,#24a878); transition:width .35s ease; }}
    .model {{ color:var(--muted); font-size:12px; margin-top:8px; word-break:break-all; }}
    .error {{ color:var(--bad); margin-top:12px; font-weight:650; white-space:pre-wrap; }}
    .results {{ display:none; margin-top:24px; }} .results.show {{ display:block; }}
    .result-head {{ display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:12px; }}
    h2 {{ margin:0; font-size:21px; color:var(--navy); }}
    .downloads {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    .downloads .button {{ font-size:13px; padding:8px 11px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:14px; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; font-size:13px; }}
    th {{ color:#fff; background:var(--navy); text-align:left; padding:11px; }}
    td {{ padding:11px; border-bottom:1px solid var(--line); vertical-align:top; line-height:1.55; }}
    tr:last-child td {{ border-bottom:0; }} tr:nth-child(even) td {{ background:#f8fbfc; }}
    .review {{ color:#a15c08; }} .verified {{ color:var(--ok); }}
    @media (max-width:680px) {{ header,.controls,.result-head {{ align-items:stretch; flex-direction:column; }} .privacy {{ white-space:normal; }} .path-row {{ grid-template-columns:1fr; }} .downloads {{ justify-content:flex-start; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>Resume Summary</h1><p class="subtitle">批量提取姓名、学历毕业时间和工作 / 实习 Title</p></div>
    <div class="header-actions"><div class="privacy">本机读 PDF · 删除联系方式 · 在线结构化抽取</div><button class="quit" id="shutdown" type="button">退出工具</button></div>
  </header>
  <section class="card">
    <div id="dropzone" tabindex="0" role="button" aria-label="拖入 PDF 简历">
      <div class="drop-title">把 PDF 简历或文件夹拖到这里</div>
      <div class="drop-help">也可以选择多个文件、选择整个文件夹，或在下方粘贴路径</div>
      <div class="actions">
        <button type="button" id="chooseFiles">选择 PDF</button>
        <button type="button" id="chooseFolder">选择文件夹</button>
      </div>
    </div>
    <input id="fileInput" type="file" accept="application/pdf,.pdf" multiple>
    <input id="folderInput" type="file" accept="application/pdf,.pdf" webkitdirectory multiple>
    <div class="path-row">
      <input id="folderPath" type="text" value="{safe_prefill}" placeholder="本机路径，例如 ~/Downloads/候选人简历">
      <button type="button" id="clearSelection">清空选择</button>
    </div>
    <div id="selection" class="selection">尚未选择文件；文件夹路径会递归读取其中的 PDF。</div>
    <div class="controls">
      <label class="check"><input id="noCache" type="checkbox"> 忽略缓存，强制重新提取</label>
      <button class="primary" id="start" type="button">开始汇总</button>
    </div>
    <div id="status" class="status" aria-live="polite">
      <div class="status-head"><span id="phase" class="phase">准备中</span><span id="detail" class="detail"></span></div>
      <div class="bar"><i id="bar"></i></div>
      <div id="model" class="model"></div>
      <div id="error" class="error"></div>
    </div>
  </section>
  <section id="results" class="card results">
    <div class="result-head"><h2>提取结果</h2><div id="downloads" class="downloads"></div></div>
    <div class="table-wrap"><table><thead><tr><th>姓名</th><th>教育经历</th><th>工作 / 实习 Title</th><th>原文件</th><th>状态</th></tr></thead><tbody id="resultBody"></tbody></table></div>
  </section>
</main>
<script>
const TOKEN = {safe_token};
let selectedFiles = [];
let activeJob = null;
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
const isPdf = file => file && (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'));

function dedupeFiles(files) {{
  const known = new Set(selectedFiles.map(f => `${{f.name}}:${{f.size}}:${{f.lastModified}}`));
  for (const file of files) {{
    const key = `${{file.name}}:${{file.size}}:${{file.lastModified}}`;
    if (isPdf(file) && !known.has(key)) {{ selectedFiles.push(file); known.add(key); }}
  }}
  updateSelection();
}}
function updateSelection() {{
  const path = $('folderPath').value.trim();
  const fileText = selectedFiles.length ? `已选择 ${{selectedFiles.length}} 个 PDF：${{selectedFiles.slice(0,4).map(f=>f.name).join('、')}}${{selectedFiles.length>4?'…':''}}` : '';
  const pathText = path ? `路径：${{path}}` : '';
  $('selection').textContent = [fileText,pathText].filter(Boolean).join('；') || '尚未选择文件；文件夹路径会递归读取其中的 PDF。';
}}
async function readEntry(entry) {{
  if (entry.isFile) return new Promise(resolve => entry.file(file => resolve(isPdf(file) ? [file] : []), () => resolve([])));
  if (!entry.isDirectory) return [];
  const reader = entry.createReader(); let children = [];
  while (true) {{ const batch = await new Promise(resolve => reader.readEntries(resolve, () => resolve([]))); if (!batch.length) break; children.push(...batch); }}
  return (await Promise.all(children.map(readEntry))).flat();
}}

$('chooseFiles').onclick = event => {{ event.stopPropagation(); $('fileInput').click(); }};
$('chooseFolder').onclick = event => {{ event.stopPropagation(); $('folderInput').click(); }};
$('fileInput').onchange = event => dedupeFiles(event.target.files);
$('folderInput').onchange = event => dedupeFiles(event.target.files);
$('folderPath').oninput = updateSelection;
$('clearSelection').onclick = () => {{ selectedFiles=[]; $('fileInput').value=''; $('folderInput').value=''; $('folderPath').value=''; updateSelection(); }};
$('dropzone').onclick = () => $('fileInput').click();
$('dropzone').onkeydown = event => {{ if (event.key === 'Enter' || event.key === ' ') $('fileInput').click(); }};
for (const eventName of ['dragenter','dragover']) $('dropzone').addEventListener(eventName, event => {{ event.preventDefault(); $('dropzone').classList.add('drag'); }});
for (const eventName of ['dragleave','drop']) $('dropzone').addEventListener(eventName, event => {{ event.preventDefault(); $('dropzone').classList.remove('drag'); }});
$('dropzone').addEventListener('drop', async event => {{
  const entries = [...event.dataTransfer.items].map(item => item.webkitGetAsEntry?.()).filter(Boolean);
  if (entries.length) dedupeFiles((await Promise.all(entries.map(readEntry))).flat());
  else dedupeFiles(event.dataTransfer.files);
}});

function formatEducation(items) {{ const labels={{bachelor:'本科',master:'硕士',phd:'博士',unknown:'学历'}}; return (items||[]).map(x=>`${{labels[x.level]||'学历'}} · ${{escapeHtml(x.school)}}${{x.graduation_date?` · ${{escapeHtml(x.graduation_date)}}`:''}}`).join('<br>') || '—'; }}
function formatJobs(items) {{ return (items||[]).map(x=>[x.company,x.title].filter(Boolean).map(escapeHtml).join(' · ')).join('<br>') || '—'; }}
function renderResults(job) {{
  $('resultBody').innerHTML = (job.records||[]).map(record => `<tr><td><strong>${{escapeHtml(record.name||'未识别')}}</strong></td><td>${{formatEducation(record.education)}}</td><td>${{formatJobs(record.employment)}}</td><td>${{escapeHtml(record.source_file)}}</td><td class="${{record.warnings?.length?'review':'verified'}}">${{record.warnings?.length?'需复核':'已核验'}}</td></tr>`).join('');
  const labels={{'resume_summary.pdf':'下载 PDF','resume_summary.html':'下载 HTML','resume_summary.csv':'下载 CSV','results.json':'下载 JSON'}};
  $('downloads').innerHTML = (job.downloads||[]).map(name=>`<a class="button" href="/downloads/${{job.id}}/${{encodeURIComponent(name)}}?token=${{encodeURIComponent(TOKEN)}}">${{labels[name]||escapeHtml(name)}}</a>`).join('') + `<button type="button" id="reveal">在 Finder 中打开</button>`;
  $('reveal').onclick = () => fetch(`/api/jobs/${{job.id}}/reveal`,{{method:'POST',headers:{{'X-Resume-Token':TOKEN}}}});
  $('results').classList.add('show');
}}
async function poll() {{
  if (!activeJob) return;
  try {{
    const response = await fetch(`/api/jobs/${{activeJob}}`, {{headers:{{'X-Resume-Token':TOKEN}}}});
    const job = await response.json();
    $('phase').textContent = job.phase || '处理中'; $('bar').style.width = `${{job.progress||0}}%`;
    $('detail').textContent = job.current_file ? `${{job.completed}} / ${{job.total}} · ${{job.current_file}}` : `${{job.completed}} / ${{job.total}}`;
    $('model').textContent = job.models?.length ? `当前免费候选：${{job.models.join(' → ')}}` : '';
    if (job.status === 'done') {{ $('start').disabled=false; activeJob=null; renderResults(job); return; }}
    if (job.status === 'error') {{ $('start').disabled=false; activeJob=null; $('error').textContent=job.error||'处理失败'; return; }}
    setTimeout(poll, 700);
  }} catch (error) {{ $('error').textContent=`无法读取进度：${{error}}`; $('start').disabled=false; activeJob=null; }}
}}
$('start').onclick = async () => {{
  if (activeJob) return;
  const path = $('folderPath').value.trim();
  if (!selectedFiles.length && !path) {{ $('status').classList.add('show'); $('error').textContent='请先选择 PDF、文件夹，或填写路径。'; return; }}
  $('status').classList.add('show'); $('results').classList.remove('show'); $('error').textContent=''; $('phase').textContent='正在提交'; $('bar').style.width='0%'; $('start').disabled=true;
  const data = new FormData(); selectedFiles.forEach(file=>data.append('files',file,file.name)); data.append('folder_path',path); if ($('noCache').checked) data.append('no_cache','1');
  try {{
    const response = await fetch('/api/jobs',{{method:'POST',headers:{{'X-Resume-Token':TOKEN}},body:data}}); const payload=await response.json();
    if (!response.ok) throw new Error(payload.error||`HTTP ${{response.status}}`);
    activeJob=payload.job_id; poll();
  }} catch (error) {{ $('error').textContent=error.message; $('start').disabled=false; }}
}};
$('shutdown').onclick = async () => {{
  if (activeJob && !window.confirm('当前仍在处理简历，确定退出吗？')) return;
  await fetch('/api/shutdown',{{method:'POST',headers:{{'X-Resume-Token':TOKEN}}}});
  document.body.innerHTML='<main><section class="card"><h2>Resume Summary 已退出</h2><p class="subtitle">现在可以关闭这个页面。</p></section></main>';
}};
updateSelection();
</script>
</body></html>'''


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume Summary 本地网页界面")
    parser.add_argument("path", nargs="?", default="", help="可选：预填 PDF 或文件夹路径")
    parser.add_argument("--port", type=int, default=0, help="监听端口；默认自动选择")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    port = args.port or _free_port()
    app = create_app(prefill_path=args.path)
    url = f"http://127.0.0.1:{port}/"
    print(f"Resume Summary 已启动：{url}")
    print("关闭本窗口即可停止服务。")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
