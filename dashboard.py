"""
Leads Dashboard — lightweight server for viewing and managing leads.
Run:  python dashboard.py        (opens http://localhost:8050)
"""

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

LEADS_DIR = Path(__file__).parent / "data" / "leads"
PORT = 8050


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default logs

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/dates":
            files = sorted(LEADS_DIR.glob("leads_*.json"), reverse=True)
            dates = []
            for f in files:
                date_str = f.stem.replace("leads_", "")
                with open(f) as fh:
                    data = json.load(fh)
                dates.append({"date": date_str, "count": data.get("total_leads", 0)})
            self._json_response(dates)

        elif parsed.path == "/api/leads":
            qs = parse_qs(parsed.query)
            date = qs.get("date", [None])[0]
            if not date:
                self._json_response({"error": "missing date param"}, 400)
                return
            fp = LEADS_DIR / f"leads_{date}.json"
            if not fp.exists():
                self._json_response({"error": "not found"}, 404)
                return
            with open(fp) as fh:
                data = json.load(fh)
            self._json_response(data)

        elif parsed.path == "/":
            self._html_response(HTML)
        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/leads":
            qs = parse_qs(parsed.query)
            date = qs.get("date", [None])[0]
            lead_id = qs.get("id", [None])[0]
            if not date or not lead_id:
                self._json_response({"error": "missing params"}, 400)
                return
            fp = LEADS_DIR / f"leads_{date}.json"
            if not fp.exists():
                self._json_response({"error": "not found"}, 404)
                return
            with open(fp) as fh:
                data = json.load(fh)
            original = len(data["leads"])
            data["leads"] = [l for l in data["leads"] if l["lead_id"] != lead_id]
            data["total_leads"] = len(data["leads"])
            if len(data["leads"]) < original:
                with open(fp, "w") as fh:
                    json.dump(data, fh, indent=2)
                self._json_response({"ok": True})
            else:
                self._json_response({"error": "lead not found"}, 404)
        else:
            self.send_error(404)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leads — CodingResearcher</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0f110f;
  --surface:#181a18;
  --surface2:#1f221f;
  --border:#2a2e2a;
  --border-hover:#3d423d;
  --text:#d4d8d0;
  --text-dim:#7a7f76;
  --text-bright:#eef0ec;
  --accent:#b8e060;
  --accent-dim:#6a8030;
  --danger:#e06050;
  --danger-bg:rgba(224,96,80,.08);
  --blue:#60a8d0;
  --orange:#d8a050;
  --serif:'DM Serif Display',Georgia,serif;
  --sans:'IBM Plex Sans','Segoe UI',sans-serif;
  --mono:'IBM Plex Mono','Fira Code',monospace;
  --radius:10px;
}
html{font-size:15px}
body{
  font-family:var(--sans);
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  line-height:1.55;
}
body::before{
  content:'';
  position:fixed;inset:0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%,rgba(184,224,96,.03),transparent),
    radial-gradient(ellipse 60% 40% at 80% 100%,rgba(96,168,208,.02),transparent);
  pointer-events:none;z-index:0;
}

/* ── Layout ── */
.shell{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:2rem 1.5rem 4rem}

/* ── Header ── */
header{
  display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:1rem;
  padding-bottom:1.5rem;
  border-bottom:1px solid var(--border);
  margin-bottom:2rem;
}
h1{font-family:var(--serif);font-size:1.8rem;color:var(--text-bright);font-weight:400;letter-spacing:-.02em}
h1 span{color:var(--accent);font-style:italic}
.subtitle{font-size:.8rem;color:var(--text-dim);font-family:var(--mono);letter-spacing:.04em;text-transform:uppercase}

/* ── Date pills ── */
.date-bar{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:2rem}
.date-pill{
  font-family:var(--mono);font-size:.82rem;
  padding:.45rem 1rem;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:999px;
  color:var(--text-dim);
  cursor:pointer;
  transition:all .2s ease;
  display:flex;align-items:center;gap:.5rem;
}
.date-pill:hover{border-color:var(--border-hover);color:var(--text)}
.date-pill.active{
  background:rgba(184,224,96,.08);
  border-color:var(--accent-dim);
  color:var(--accent);
}
.date-pill .count{
  font-size:.7rem;
  background:var(--surface2);
  padding:.1rem .45rem;
  border-radius:999px;
  min-width:1.3rem;text-align:center;
}
.date-pill.active .count{background:rgba(184,224,96,.15)}

/* ── Empty state ── */
.empty{
  text-align:center;padding:5rem 1rem;
  color:var(--text-dim);font-family:var(--mono);font-size:.9rem;
}
.empty b{display:block;font-size:1.6rem;margin-bottom:.5rem;font-family:var(--serif);color:var(--text);font-weight:400}

/* ── Cards grid ── */
.grid{display:grid;grid-template-columns:1fr;gap:1rem}
@media(min-width:720px){.grid{grid-template-columns:1fr 1fr}}

/* ── Card ── */
.card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.4rem 1.5rem;
  transition:border-color .25s, transform .25s, opacity .4s;
  position:relative;
  overflow:hidden;
}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--blue));
  opacity:0;transition:opacity .25s;
}
.card:hover{border-color:var(--border-hover);transform:translateY(-2px)}
.card:hover::before{opacity:1}
.card.removing{opacity:0;transform:scale(.96) translateY(8px)}

/* card header */
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:.75rem;margin-bottom:1rem}
.card-institution{font-family:var(--serif);font-size:1.15rem;color:var(--text-bright);line-height:1.3}
.card-rank{
  font-family:var(--mono);font-size:.7rem;font-weight:600;
  background:var(--surface2);color:var(--accent);
  padding:.2rem .55rem;border-radius:999px;
  white-space:nowrap;flex-shrink:0;
}

/* meta row */
.meta{display:flex;flex-wrap:wrap;gap:.4rem .8rem;margin-bottom:1rem}
.tag{
  font-family:var(--mono);font-size:.72rem;
  padding:.2rem .55rem;border-radius:4px;
  background:var(--surface2);color:var(--text-dim);
  display:inline-flex;align-items:center;gap:.3rem;
}
.tag .icon{font-size:.8rem}
.tag.country{color:var(--blue)}
.tag.service{color:var(--orange)}
.tag.grant{color:var(--accent)}
.tag.signal{color:#c090e0}
.tag.hiring{color:#e0a0b0}

/* score bar */
.score-section{margin-bottom:1rem}
.score-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.35rem}
.score-label{font-family:var(--mono);font-size:.7rem;color:var(--text-dim);width:4.5rem;text-align:right;flex-shrink:0}
.score-track{flex:1;height:6px;background:var(--surface2);border-radius:3px;overflow:hidden;position:relative}
.score-fill{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.22,1,.36,1)}
.score-val{font-family:var(--mono);font-size:.7rem;color:var(--text-dim);width:2rem;flex-shrink:0}
.score-total{
  display:inline-flex;align-items:baseline;gap:.3rem;
  font-family:var(--mono);font-size:.82rem;color:var(--text);margin-bottom:.6rem;
}
.score-total b{font-size:1.1rem;color:var(--accent);font-weight:600}

/* pain & justifications */
.detail-block{margin-bottom:.8rem}
.detail-title{font-family:var(--mono);font-size:.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
.detail-text{font-size:.82rem;color:var(--text);line-height:1.5}

/* outreach section */
.outreach{
  margin-top:1rem;padding-top:1rem;
  border-top:1px solid var(--border);
}
.outreach-toggle{
  font-family:var(--mono);font-size:.75rem;
  color:var(--accent-dim);cursor:pointer;
  background:none;border:none;
  display:flex;align-items:center;gap:.3rem;
  transition:color .2s;
}
.outreach-toggle:hover{color:var(--accent)}
.outreach-toggle .arrow{transition:transform .2s;display:inline-block}
.outreach-toggle.open .arrow{transform:rotate(90deg)}
.outreach-body{
  max-height:0;overflow:hidden;
  transition:max-height .35s ease, padding .35s ease;
}
.outreach-body.open{max-height:600px;padding-top:.8rem}
.outreach-field{margin-bottom:.7rem}
.outreach-field-label{font-family:var(--mono);font-size:.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.15rem}
.outreach-field-text{font-size:.82rem;line-height:1.5;color:var(--text)}
.email-block{
  background:var(--surface2);
  border:1px solid var(--border);
  border-radius:6px;
  padding:.8rem 1rem;
  font-size:.8rem;line-height:1.6;
  color:var(--text);
  white-space:pre-wrap;
  position:relative;
}
.copy-btn{
  position:absolute;top:.5rem;right:.5rem;
  background:var(--surface);border:1px solid var(--border);
  color:var(--text-dim);font-family:var(--mono);font-size:.65rem;
  padding:.2rem .5rem;border-radius:4px;cursor:pointer;
  transition:all .2s;
}
.copy-btn:hover{border-color:var(--accent-dim);color:var(--accent)}

/* delete button */
.delete-btn{
  margin-top:1rem;
  background:var(--danger-bg);
  border:1px solid rgba(224,96,80,.2);
  color:var(--danger);
  font-family:var(--mono);font-size:.75rem;
  padding:.4rem .9rem;border-radius:6px;
  cursor:pointer;
  transition:all .2s;
  display:inline-flex;align-items:center;gap:.35rem;
}
.delete-btn:hover{background:rgba(224,96,80,.15);border-color:rgba(224,96,80,.4)}

/* confirm overlay */
.confirm-overlay{
  position:absolute;inset:0;
  background:rgba(15,17,15,.92);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:.8rem;z-index:2;
  opacity:0;pointer-events:none;
  transition:opacity .2s;
  border-radius:var(--radius);
}
.confirm-overlay.visible{opacity:1;pointer-events:auto}
.confirm-text{font-family:var(--sans);font-size:.9rem;color:var(--text-bright)}
.confirm-actions{display:flex;gap:.5rem}
.confirm-actions button{
  font-family:var(--mono);font-size:.78rem;
  padding:.4rem 1rem;border-radius:6px;cursor:pointer;
  border:1px solid;transition:all .15s;
}
.btn-yes{background:var(--danger);border-color:var(--danger);color:#fff}
.btn-yes:hover{background:#c84838}
.btn-no{background:transparent;border-color:var(--border);color:var(--text)}
.btn-no:hover{border-color:var(--text-dim)}

/* loading */
.loading{text-align:center;padding:3rem;color:var(--text-dim);font-family:var(--mono);font-size:.85rem}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
.loading::after{content:'';display:inline-block;width:6px;height:6px;background:var(--accent);border-radius:50%;margin-left:.5rem;animation:pulse 1s infinite}

/* fade-in for cards */
@keyframes cardIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{animation:cardIn .4s ease both}
.card:nth-child(2){animation-delay:.06s}
.card:nth-child(3){animation-delay:.12s}
.card:nth-child(4){animation-delay:.18s}
.card:nth-child(5){animation-delay:.24s}
.card:nth-child(6){animation-delay:.3s}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <h1>Lead <span>Scout</span></h1>
      <div class="subtitle">CodingResearcher pipeline</div>
    </div>
  </header>
  <div class="date-bar" id="dateBar"></div>
  <div id="content"><div class="loading">Loading dates</div></div>
</div>

<script>
const $ = s => document.querySelector(s);
let currentDate = null;

const SCORE_COLORS = {
  pain:'#e06050',
  budget:'#b8e060',
  timing:'#60a8d0',
  fit:'#d8a050',
};

function flag(code) {
  if (!code) return '';
  const cp = [...code.toUpperCase()].map(c => 0x1F1E6 + c.charCodeAt(0) - 65);
  return cp.map(c => String.fromCodePoint(c)).join('');
}

async function init() {
  const res = await fetch('/api/dates');
  const dates = await res.json();
  renderDates(dates);
  if (dates.length && dates[0].count > 0) selectDate(dates[0].date);
  else if (dates.length) selectDate(dates[0].date);
  else $('#content').innerHTML = '<div class="empty"><b>No leads yet</b>Run the pipeline to generate leads.</div>';
}

function renderDates(dates) {
  $('#dateBar').innerHTML = dates.map(d =>
    `<button class="date-pill" data-date="${d.date}">
      ${d.date}<span class="count">${d.count}</span>
    </button>`
  ).join('');
  $('#dateBar').querySelectorAll('.date-pill').forEach(btn =>
    btn.addEventListener('click', () => selectDate(btn.dataset.date))
  );
}

async function selectDate(date) {
  currentDate = date;
  document.querySelectorAll('.date-pill').forEach(p =>
    p.classList.toggle('active', p.dataset.date === date)
  );
  $('#content').innerHTML = '<div class="loading">Loading leads</div>';
  const res = await fetch(`/api/leads?date=${date}`);
  const data = await res.json();
  renderLeads(data);
}

function scoreBar(label, value, max, color) {
  const pct = Math.min(100, (value / max) * 100);
  return `<div class="score-row">
    <span class="score-label">${label}</span>
    <div class="score-track"><div class="score-fill" style="width:${pct}%;background:${color}"></div></div>
    <span class="score-val">${value.toFixed(1)}</span>
  </div>`;
}

function renderLeads(data) {
  if (!data.leads || data.leads.length === 0) {
    $('#content').innerHTML = '<div class="empty"><b>No leads</b>No leads were found for this date.</div>';
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'grid';
  data.leads.forEach(lead => {
    const s = lead.score || {};
    const loc = [lead.city, lead.country].filter(Boolean).join(', ');
    const grants = (lead.active_eu_grants || []);
    const hiring = (lead.hiring_signals || []);
    const signals = (lead.source_signals || []);
    const pains = (lead.pain_points || []);
    const o = lead.outreach || {};

    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.id = lead.lead_id;
    card.innerHTML = `
      <div class="confirm-overlay" id="confirm-${lead.lead_id}">
        <div class="confirm-text">Remove this lead?</div>
        <div class="confirm-actions">
          <button class="btn-yes" data-id="${lead.lead_id}">Remove</button>
          <button class="btn-no" data-id="${lead.lead_id}">Cancel</button>
        </div>
      </div>
      <div class="card-head">
        <div class="card-institution">${lead.institution_name}</div>
        <div class="card-rank">#${lead.rank}</div>
      </div>
      <div class="meta">
        ${loc ? `<span class="tag country"><span class="icon">${flag(lead.country)}</span>${loc}</span>` : ''}
        ${lead.service_match ? `<span class="tag service">${lead.service_match.replace(/_/g,' ')}</span>` : ''}
        ${grants.map(g => `<span class="tag grant">EU ${g}</span>`).join('')}
        ${signals.map(s => `<span class="tag signal">${s.replace(/_/g,' ')}</span>`).join('')}
        ${hiring.map(h => `<span class="tag hiring">${h.length > 40 ? h.slice(0,38)+'…' : h}</span>`).join('')}
        ${pains.map(p => `<span class="tag">${p.replace(/_/g,' ')}</span>`).join('')}
      </div>
      <div class="score-section">
        <div class="score-total"><b>${(s.total||0).toFixed(1)}</b>/10</div>
        ${scoreBar('Pain', s.pain_intensity||0, 10, SCORE_COLORS.pain)}
        ${scoreBar('Budget', s.budget_signal||0, 10, SCORE_COLORS.budget)}
        ${scoreBar('Timing', s.timing_signal||0, 10, SCORE_COLORS.timing)}
        ${scoreBar('Fit', s.fit_score||0, 10, SCORE_COLORS.fit)}
      </div>
      ${s.pain_justification ? `<div class="detail-block"><div class="detail-title">Pain</div><div class="detail-text">${s.pain_justification}</div></div>` : ''}
      ${s.budget_justification ? `<div class="detail-block"><div class="detail-title">Budget</div><div class="detail-text">${s.budget_justification}</div></div>` : ''}
      ${s.timing_justification ? `<div class="detail-block"><div class="detail-title">Timing</div><div class="detail-text">${s.timing_justification}</div></div>` : ''}
      ${s.fit_justification ? `<div class="detail-block"><div class="detail-title">Fit</div><div class="detail-text">${s.fit_justification}</div></div>` : ''}
      ${o.hook ? `
      <div class="outreach">
        <button class="outreach-toggle" onclick="toggleOutreach(this)">
          <span class="arrow">&#9654;</span> Outreach
        </button>
        <div class="outreach-body">
          <div class="outreach-field"><div class="outreach-field-label">Hook</div><div class="outreach-field-text">${o.hook}</div></div>
          <div class="outreach-field"><div class="outreach-field-label">Pitch</div><div class="outreach-field-text">${o.service_pitch||''}</div></div>
          <div class="outreach-field"><div class="outreach-field-label">Free Audit</div><div class="outreach-field-text">${o.free_audit_offer||''}</div></div>
          ${o.email_draft ? `
          <div class="outreach-field">
            <div class="outreach-field-label">Email Draft</div>
            <div class="email-block">${o.email_draft}<button class="copy-btn" onclick="copyEmail(this, event)">copy</button></div>
          </div>` : ''}
        </div>
      </div>` : ''}
      <button class="delete-btn" onclick="askDelete('${lead.lead_id}')">&#10005; Remove lead</button>
    `;
    grid.appendChild(card);
  });
  $('#content').innerHTML = '';
  $('#content').appendChild(grid);

  // bind confirm buttons
  grid.querySelectorAll('.btn-yes').forEach(b =>
    b.addEventListener('click', () => doDelete(b.dataset.id))
  );
  grid.querySelectorAll('.btn-no').forEach(b =>
    b.addEventListener('click', () => {
      document.getElementById('confirm-' + b.dataset.id).classList.remove('visible');
    })
  );
}

function toggleOutreach(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

function askDelete(id) {
  document.getElementById('confirm-' + id).classList.add('visible');
}

async function doDelete(id) {
  const card = document.querySelector(`.card[data-id="${id}"]`);
  card.classList.add('removing');
  await fetch(`/api/leads?date=${currentDate}&id=${id}`, { method: 'DELETE' });
  setTimeout(() => selectDate(currentDate), 400);
}

function copyEmail(btn, e) {
  e.stopPropagation();
  const text = btn.parentElement.textContent.replace('copy','').trim();
  navigator.clipboard.writeText(text);
  btn.textContent = 'copied!';
  setTimeout(() => btn.textContent = 'copy', 1500);
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"  Leads dashboard → {url}")
    print("  Ctrl+C to stop\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
