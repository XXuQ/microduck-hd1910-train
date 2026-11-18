#!/usr/bin/env python3
"""训练监控面板 —— 起一个网页看实时训练曲线。

数据优先读 **tfevents**（rsl_rl 每轮都写，实时且准确），读不到再退回解析
stdout 日志。除了 tensorboard 的 EventAccumulator（引擎 venv 里自带）之外
不依赖任何三方包，也不碰 wandb。

用法（在训练机上，用引擎的 venv 跑）：

    cd <引擎目录>
    uv run python <仓库>/tools/monitor_server.py --port 8080

浏览器怎么连（训练机是远程时）：

    ssh -L 8080:localhost:8080 -p 端口 root@主机
    # 本地打开 http://localhost:8080

训练在 WSL、浏览器在同一个 Windows 上：直接开 http://localhost:8080

选项：
    --engine PATH   引擎目录（默认读 .engine_path）
    --run NAME      只看某个 run（默认最新那个）
    --total N       总轮数（tfevents 里没有，用来画进度条；不填就自动找）
    --log PATH      强制用文本日志模式，可重复
    --port/--host/--refresh
"""

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PREFERRED = [
    "总奖励",
    "Metrics/twist/error_vel_xy",
    "Episode_Termination/fell_over",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/air_time",
    "Logs/学习率",
]


# ---------------------------------------------------------------- 引擎/数据源
def resolve_engine(arg):
    if arg:
        return Path(arg)
    f = REPO / ".engine_path"
    if f.is_file():
        return Path(f.read_text().strip())
    for c in (REPO.parent / "microduck_rl", Path.home() / "microduck_rl",
              Path.home() / "ducktest" / "microduck_rl"):
        if (c / ".git").is_dir():
            return c
    return None


def find_runs(engine):
    """返回 [(run_dir_name, tfevents_path, mtime)]，按时间倒序。"""
    if not engine:
        return []
    out = []
    for p in engine.glob("logs/rsl_rl/*/*/events.out.tfevents.*"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((p.parent.name, p, st.st_mtime))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


class TfSource:
    """包一层 EventAccumulator，按文件大小变化决定是否 Reload。"""

    def __init__(self, path):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        self.path = Path(path)
        self.ea = EventAccumulator(str(path), size_guidance={"scalars": 3000})
        self.ea.Reload()
        self._size = self._cur_size()
        self._cache = None

    def _cur_size(self):
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def scalars(self):
        size = self._cur_size()
        if size != self._size:
            try:
                self.ea.Reload()
                self._size = size
                self._cache = None
            except Exception:
                pass
        if self._cache is not None:
            return self._cache
        out = {}
        try:
            tags = self.ea.Tags()["scalars"]
        except Exception:
            tags = []
        for t in tags:
            try:
                ev = self.ea.Scalars(t)
            except Exception:
                continue
            if ev:
                out[t] = [(e.step, e.value) for e in ev]
        # 总奖励 = 所有 Episode_Reward/* 逐轮求和（和训练打印的 Mean reward 一致）
        rew_keys = [k for k in out if k.startswith("Episode_Reward/")]
        if rew_keys:
            n = min(len(out[k]) for k in rew_keys)
            if n > 0:
                steps = [out[rew_keys[0]][i][0] for i in range(n)]
                sums = [sum(out[k][i][1] for k in rew_keys) for i in range(n)]
                out["总奖励"] = list(zip(steps, sums))
        self._cache = out
        return out


def read_total_from_logs(engine, run_name):
    """从同名 stdout 日志里捞 `Learning iteration N/T` 的 T（tfevents 里没有总轮数）。"""
    pat = re.compile(r"Learning iteration\s+\d+\s*/\s*(\d+)")
    dirs = [Path(Handler.logs_dir)] if Handler.logs_dir else []
    dirs += [REPO / "logs", Path.cwd() / "logs", Path("/tmp")]
    seen, cands = set(), []
    for d in dirs:
        if not d or not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p not in seen:
                seen.add(p)
                cands.append(p)
    best = None
    for p in cands[:20]:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if run_name not in p.name and run_name not in txt[:200000]:
            continue
        m = pat.findall(txt)
        if m:
            best = int(m[-1])
    return best


def downsample(pts, max_points=800):
    if len(pts) <= max_points:
        return pts
    step = len(pts) / max_points
    return [pts[int(i * step)] for i in range(max_points)]


# ---------------------------------------------------------------- 文本日志兜底
ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
METRIC_RE = re.compile(r"^\s+([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)+):\s+(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*$")
REWARD_RE = re.compile(r"^\s+Mean reward:\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$")


def parse_text_log(path):
    series, cur, total = {}, None, None
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    cur, total = int(m.group(1)), int(m.group(2))
                    continue
                if cur is None:
                    continue
                mr = REWARD_RE.match(line)
                if mr:
                    series.setdefault("总奖励", []).append((cur, float(mr.group(1))))
                    continue
                mm = METRIC_RE.match(line)
                if mm:
                    series.setdefault(mm.group(1), []).append((cur, float(mm.group(2))))
    except OSError:
        pass
    return series, cur, total


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    engine = None
    runs = []
    refresh = 5
    forced_total = None
    text_logs = []
    logs_dir = None
    auto_runs = False
    _src_cache = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self.handle_state()
        else:
            self._send(200, PAGE.replace("__REFRESH__", str(self.refresh)), "text/html; charset=utf-8")

    def _run_by_name(self, name):
        for run_name, path, _ in self.runs:
            if run_name == name:
                return run_name, path
        return (self.runs[0][0], self.runs[0][1]) if self.runs else (None, None)

    def handle_state(self):
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        want = (q.get("run") or [""])[0]

        # run 列表每次请求重扫：训练随时会新建 run，缓存住就看不到新任务
        if getattr(self, "auto_runs", False):
            self.runs = find_runs(self.engine)

        payload = {"runs": [{"name": n, "mtime": t} for n, _, t in self.runs],
                   "checkpoints": list_checkpoints(self.engine)}

        if self.runs:
            run_name, path = self._run_by_name(want)
            key = str(path)
            src = self._src_cache.get(key)
            try:
                if src is None or getattr(src, "path", None) != path:
                    src = TfSource(path)
                    self._src_cache[key] = src
                series = src.scalars()
            except Exception as e:  # tensorboard 缺失或文件半写
                payload["error"] = f"读 tfevents 失败：{e}"
                series, run_name, path = {}, run_name, path
            payload["source"] = "tfevents"
            payload["current"] = run_name
        else:
            base = Path(self.logs_dir) if self.logs_dir else (REPO / "logs")
            logs = self.text_logs or sorted(base.glob("*.log"),
                                            key=lambda p: p.stat().st_mtime, reverse=True)[:8]
            if not logs:
                payload["error"] = "没找到 tfevents，也没有 logs/*.log；训练起来后刷新即可"
                self._send(200, json.dumps(payload))
                return
            chosen = Path(logs[0])
            for p in logs:
                if want and p.name == want:
                    chosen = p
            series, cur, total = parse_text_log(chosen)
            payload.update({"source": "text", "current": chosen.name, "iter": cur, "total": total})
            payload["runs"] = [{"name": p.name, "mtime": p.stat().st_mtime} for p in logs]

        series = {k: downsample(v) for k, v in series.items() if v}
        payload["series"] = series
        steps = [pts[-1][0] for pts in series.values() if pts]
        payload["iter"] = max(steps) if steps else None

        total = self.forced_total
        if not total and payload.get("current"):
            total = read_total_from_logs(self.engine, payload["current"])
        payload["total"] = total or payload.get("total")
        payload["now"] = time.time()

        keys = [k for k in PREFERRED if k in series]
        for k in sorted(series):
            if k not in keys and len(series[k]) > 1:
                keys.append(k)
        payload["keys"] = keys
        self._send(200, json.dumps(payload))


def list_checkpoints(engine):
    if not engine:
        return []
    out = []
    for p in engine.glob("logs/rsl_rl/*/*/model_*.pt"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "run": p.parent.name,
                    "mb": round(st.st_size / 1048576, 1),
                    "time": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)),
                    "mtime": st.st_mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:20]


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duck 训练监控</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#242a35;--fg:#e6e9ef;--dim:#8b94a7;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:14px;padding:12px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:9;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
select,button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font-size:13px;max-width:340px}
#dot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex:none}
.wrap{padding:16px 20px 40px;max-width:1320px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:10px;margin-bottom:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 14px}
.card .k{color:var(--dim);font-size:12px;margin-bottom:3px}
.card .v{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums}
.bar{height:6px;background:#222836;border-radius:99px;overflow:hidden;margin:6px 0 16px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);width:0;transition:width .4s}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px 6px}
.chart h3{margin:2px 0 8px;font-size:13px;font-weight:600;display:flex;justify-content:space-between;gap:10px}
.chart h3 span{color:var(--dim);font-weight:400;font-variant-numeric:tabular-nums}
canvas{width:100%;height:180px;display:block}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
tr:hover td{background:#1b1f28}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.hint{color:var(--dim);font-size:12px;margin-top:12px}
</style></head><body>
<header>
  <h1>Duck 训练监控</h1><div id="dot"></div>
  <select id="runsel"></select>
  <select id="more" title="选择要看的指标"></select>
  <span class="mono" id="updated" style="color:#8b94a7;font-size:12px">--</span>
  <button id="pause">暂停</button>
</header>
<div class="wrap">
  <div class="cards">
    <div class="card"><div class="k">轮次</div><div class="v" id="c-iter">--</div></div>
    <div class="card"><div class="k">总奖励</div><div class="v" id="c-rew">--</div></div>
    <div class="card"><div class="k">error_vel_xy</div><div class="v" id="c-vel">--</div></div>
    <div class="card"><div class="k">摔倒率</div><div class="v" id="c-fell">--</div></div>
    <div class="card"><div class="k">总步数/s</div><div class="v" id="c-fps">--</div></div>
    <div class="card"><div class="k">数据源</div><div class="v" id="c-src" style="font-size:15px">--</div></div>
  </div>
  <div class="bar"><i id="prog"></i></div>

  <div class="grid">
    <div class="chart"><h3>总奖励 <span id="v-rew"></span></h3><canvas id="ch-rew"></canvas></div>
    <div class="chart"><h3>error_vel_xy（越低越好，目标 ≤0.30）<span id="v-vel"></span></h3><canvas id="ch-vel"></canvas></div>
    <div class="chart"><h3>摔倒率 % <span id="v-fell"></span></h3><canvas id="ch-fell"></canvas></div>
    <div class="chart"><h3 id="more-title">更多指标 <span id="v-more"></span></h3><canvas id="ch-more"></canvas></div>
  </div>

  <div class="chart" style="margin-top:14px">
    <h3>Checkpoint <span id="ck-count"></span></h3>
    <table><thead><tr><th>文件</th><th>run</th><th>大小</th><th>时间</th></tr></thead><tbody id="cks"></tbody></table>
  </div>
  <div class="hint">数据来自 tfevents（rsl_rl 每轮写入）；读不到时退回解析 stdout 日志。页面每 __REFRESH__ 秒刷新，曲线降采样到 800 点。</div>
</div>
<script>
let cur=null, paused=false, moreKey=null, runName=null;
const $=id=>document.getElementById(id);
const fmt=(v,d=3)=>v==null?'--':(Math.abs(v)>=100?v.toFixed(0):v.toFixed(d));
const last=a=>a&&a.length?a[a.length-1][1]:null;

function draw(cv, pts, color, target){
  const dpr=devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr;
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  if(!pts||pts.length<2){g.fillStyle='#8b94a7';g.font='12px sans-serif';g.fillText('等数据…',10,20);return}
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1===x0)x1=x0+1; if(y1===y0){y1=y0+1;y0-=1}
  const pad=8, W=w-pad*2, H=h-pad*2-6;
  const X=x=>pad+(x-x0)/(x1-x0)*W, Y=y=>pad+H-(y-y0)/(y1-y0)*H;
  g.strokeStyle='#242a35'; g.lineWidth=1;
  for(let i=0;i<=4;i++){const y=pad+H*i/4;g.beginPath();g.moveTo(pad,y);g.lineTo(pad+W,y);g.stroke()}
  if(target!=null && target>=y0 && target<=y1){
    g.strokeStyle='#d29922';g.setLineDash([4,4]);g.beginPath();
    g.moveTo(pad,Y(target));g.lineTo(pad+W,Y(target));g.stroke();g.setLineDash([]);
  }
  g.strokeStyle=color;g.lineWidth=1.8;g.beginPath();
  pts.forEach((p,i)=> i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
  g.stroke();
  g.fillStyle='#8b94a7';g.font='11px ui-monospace,monospace';
  g.fillText(y1.toFixed(2),pad+2,pad+10); g.fillText(y0.toFixed(2),pad+2,pad+H-2);
  g.fillText(String(x1),pad+W-44,pad+H-2);
}

function render(){
  if(!cur) return;
  const s=cur.series||{};
  const rw=s['总奖励'], ve=s['Metrics/twist/error_vel_xy'], fe=s['Episode_Termination/fell_over'], fp=s['Perf/total_fps'];
  const it=cur.iter||0, tot=cur.total||0;
  $('c-iter').textContent = it? (it+(tot?' / '+tot:'')) : '--';
  $('prog').style.width = tot? Math.min(100,it/tot*100)+'%':'0';
  $('c-rew').textContent=fmt(last(rw),2); $('v-rew').textContent=fmt(last(rw),2);
  $('c-vel').textContent=fmt(last(ve),3); $('v-vel').textContent=fmt(last(ve),3);
  $('c-fell').textContent=last(fe)==null?'--':fmt(last(fe),1)+'%'; $('v-fell').textContent=fmt(last(fe),1);
  $('c-fps').textContent=last(fp)?Math.round(last(fp)).toLocaleString():'--';
  $('c-src').textContent=cur.source==='tfevents'?'tfevents':(cur.source||'--');
  $('updated').textContent='更新 '+new Date().toLocaleTimeString()+ (cur.error?(' · '+cur.error):'');
  draw($('ch-rew'), rw, '#58a6ff');
  draw($('ch-vel'), ve, '#f85149', 0.30);
  draw($('ch-fell'), fe, '#d29922');
  if(moreKey){ $('more-title').firstChild.textContent=moreKey+' ';
    draw($('ch-more'), s[moreKey], '#3fb950'); $('v-more').textContent=fmt(last(s[moreKey]),4); }

  const cks=cur.checkpoints||[];
  $('ck-count').textContent=cks.length?cks.length+' 个（最近 20）':'';
  $('cks').innerHTML=cks.map(c=>`<tr><td class="mono">${c.name}</td><td class="mono" style="color:#8b94a7">${c.run}</td><td>${c.mb} MB</td><td>${c.time}</td></tr>`).join('')
    ||'<tr><td colspan="4" style="color:#8b94a7">还没有 checkpoint</td></tr>';
}

function fillSelectors(){
  const rs=$('runsel'), runs=(cur.runs||[]);
  if(rs.options.length!==runs.length){
    rs.innerHTML=runs.map(r=>`<option value="${r.name}">${r.name}</option>`).join('');
  }
  if(cur.current) rs.value=cur.current;
  if(!runName) runName=cur.current;

  const ms=$('more'), keys=cur.keys||[];
  if(ms.options.length!==keys.length){
    ms.innerHTML=keys.map(k=>`<option value="${k}">${k}</option>`).join('');
    const skip=['总奖励','Metrics/twist/error_vel_xy','Episode_Termination/fell_over'];
    moreKey=keys.find(k=>!skip.includes(k))||keys[0]||null;
    if(moreKey) ms.value=moreKey;
  }
}

async function tick(){
  if(paused) return;
  try{
    const q=runName?('?run='+encodeURIComponent(runName)):'';
    const d=await (await fetch('/api/state'+q,{cache:'no-store'})).json();
    $('dot').style.background=d.error?'#d29922':'#3fb950';
    cur=d; fillSelectors(); render();
  }catch(e){ $('dot').style.background='#f85149'; $('updated').textContent='连接断开'; }
}

$('runsel').addEventListener('change',e=>{ runName=e.target.value; cur=null; tick(); });
$('more').addEventListener('change',e=>{ moreKey=e.target.value; render(); });
$('pause').addEventListener('click',e=>{ paused=!paused; e.target.textContent=paused?'继续':'暂停'; if(!paused) tick(); });
tick(); setInterval(tick, __REFRESH__*1000);
addEventListener('resize',()=>render());
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--logs-dir", default=None,
                    help="stdout 日志目录（用来识别总轮数、文本模式兜底）")
    ap.add_argument("--log", action="append", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--refresh", type=int, default=5)
    a = ap.parse_args()

    engine = resolve_engine(a.engine)
    runs = [] if a.log else find_runs(engine)

    Handler.engine = engine
    Handler.runs = runs
    Handler.auto_runs = not a.log   # 没指定日志＝自动发现，每次请求重扫 run
    Handler.refresh = a.refresh
    Handler.forced_total = a.total
    Handler.text_logs = [Path(p) for p in a.log] if a.log else []
    Handler.logs_dir = a.logs_dir
    Handler.run_wanted = a.run

    print(f"监控地址 : http://{a.host}:{a.port}")
    print(f"引擎目录 : {engine or '（未找到）'}")
    if runs:
        print(f"发现 {len(runs)} 个 run，默认看最新：{runs[0][0]}")
        for n, p, _ in runs[:5]:
            print(f"  - {n}")
    else:
        print("⚠ 没找到 tfevents；将退回解析 仓库/logs/*.log")
    print()
    print("远程机器访问： ssh -L %d:localhost:%d user@host   然后本地开 http://localhost:%d"
          % (a.port, a.port, a.port))
    if a.host == "0.0.0.0":
        print("⚠ 绑定 0.0.0.0：同网段都能访问，公网机器请用 SSH 隧道")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
