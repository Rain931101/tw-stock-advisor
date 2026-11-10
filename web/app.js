// 台股選股顧問前端（無建置步驟的原生 JS）
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toLocaleString("zh-TW", { maximumFractionDigits: d, minimumFractionDigits: 0 });
const pct = (v) => v === null || v === undefined ? "—" : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${fmt(v)}%</span>`;
const signed = (v, d = 0) => v === null || v === undefined ? "—" : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${fmt(v, d)}</span>`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) { const t = await r.text(); throw new Error(t); }
  return r.json();
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); setTimeout(() => t.classList.add("hidden"), 2600); }

// ---------------- 分頁 ----------------
const loaders = {};
let current = "market";
function show(tab) {
  current = tab;
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $$(".tab").forEach((s) => s.classList.toggle("on", s.id === `tab-${tab}`));
  loaders[tab]?.();
  history.replaceState(null, "", `#${tab}`);
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

setInterval(() => { $("#clock").textContent = new Date().toLocaleString("zh-TW", { hour12: false, timeZone: "Asia/Taipei" }); }, 1000);

// ---------------- 大盤 ----------------
async function loadForecast() {
  const el = $("#forecast");
  let f;
  try { f = window.__fc = await api("/api/market/forecast"); } catch (e) { el.innerHTML = `<h3>明日大盤預測</h3><p class="muted">${esc(e.message)}</p>`; return; }
  const p = Math.round(f.prob_up * 100), cls = f.prob_up >= 0.6 ? "up" : f.prob_up <= 0.4 ? "down" : "";
  const n = f.night, r = f.range, o = f.model.oos;
  const us = f.us;
  const src = (n ? `已納入${esc(n.source)} ${n.night_ret > 0 ? "+" : ""}${n.night_ret}%（${esc(n.time)}）` : "收盤後預測（夜盤 15:00 開盤後會自動更新，準確率較高）")
    + (us ? `＋美股（${us.date} 收盤）` : n ? "｜美股收盤（清晨 4～5 點）後會再更新" : "");
  const op = f.open;
  el.innerHTML = `<h3>明日大盤預測 <span class="muted">依 ${f.date} 收盤 ${fmt(f.close)}｜${src}</span></h3>
    <div class="grid tiles">
      <div class="tile"><div class="k">明天收漲機率</div><div class="v ${cls}">${p}%</div><div class="muted">${esc(f.lean)}</div></div>
      ${op ? `<div class="tile"><div class="k">預估開盤</div><div class="v ${op.pct > 0 ? "up" : op.pct < 0 ? "down" : ""}">${fmt(op.price, 0)}</div><div class="muted">${pct(op.pct)}｜開高／開低方向歷史準確 ${Math.round((op.dir_acc || 0) * 100)}%，平均誤差 ±${Number(op.mae).toFixed(2)}%</div></div>` : ""}
      <div class="tile"><div class="k">預估收盤（中心）</div><div class="v">${fmt(f.center, 0)}</div><div class="muted">${pct((f.center / f.close - 1) * 100)}</div></div>
      <div class="tile"><div class="k">預估區間（中位數）</div><div class="v">${fmt(r.low, 0)}～${fmt(r.high, 0)}</div><div class="muted">較寬區間 ${fmt(r.low10, 0)}～${fmt(r.high90, 0)}（歷史上約 ${Math.round(r.coverage80 * 100)}% 的日子落在內）</div></div>
    </div>
    ${us ? `<p>美股：${Object.entries(us).filter(([k]) => k !== "date").map(([k, v]) => `<span class="chip ${v > 0 ? "up" : v < 0 ? "down" : ""}">${esc(k)} ${v > 0 ? "+" : ""}${v}%</span>`).join("")}</p>` : ""}
    ${f.reasons.length ? `<p>主要原因：${f.reasons.map((x) => `<span class="chip ${x.effect === "偏多" ? "up" : "down"}">${esc(x.feature)} ${x.value > 0 ? "+" : ""}${x.value}｜${x.effect}</span>`).join("")}</p>` : ""}
    <div id="gapfade"></div>
    <p class="muted">${esc(f.note)}。${f.mode === "close" ? "收盤當下沒有夜盤與美股資訊，方向預測和猜漲差不多，請以區間為主參考；" : ""}模型只用預測當下已知的資料（台股收盤、夜盤、美股），以走勢前進方式在沒看過的期間驗證（${o.from} 起 ${o.n} 天）。僅為統計預測，不是投資建議。</p>`;
}
async function loadGapfade(f) {
  const g = await api("/api/market/gapfade");
  const pm = g.premarket, st = g.stats || {}, ps = st.premarket, it = st.intraday;
  const P = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
  const gapPct = f ? (f.open ? f.open.pct : (f.center / f.close - 1) * 100) : 0;
  const gapUp = gapPct >= 0.5;
  let html = `<h4>開高走低判斷（講義〈判斷大盤〉）</h4>`;
  if (g.today) html += `<p class="warn"><b>⚠️ 今天已觸發開高走低警示：</b>開高 ${g.today.gap}% 後跌破開盤價（${g.today.check}，${g.today.drop_from_open}%）</p>`;
  if (gapUp) html += `<p>預估明天${f.mode === "us" ? "依夜盤與美股" : f.mode === "night" ? "依夜盤" : ""}開高約 ${pct(gapPct)}：開盤後留意是否跌破開盤價。</p>`;
  if (pm && ps) {
    html += `<ul class="list">
      <li>${pm.A ? "🔴" : "⚪"} 正價差擴大＋外資期貨空單增加：${pm.A ? "<b>成立</b>" : "未成立"}
        <span class="muted">（價差 ${pm.basis_prev ?? "—"} → ${pm.basis ?? "—"}、外資空單 ${pm.short_chg > 0 ? "+" : ""}${pm.short_chg ?? "—"} 口）｜歷史：成立後開高的日子收盤低於開盤 ${P(ps.A.fade)}（${ps.A.n} 天），所有開高日 ${P(ps.base.fade)}</span></li>
      <li>${pm.B ? "🔴" : "⚪"} 外資 BP 增加 > BC 增加：${pm.B ? "<b>成立</b>" : "未成立"}
        <span class="muted">（BC ${pm.bc_chg > 0 ? "+" : ""}${pm.bc_chg ?? "—"}、BP ${pm.bp_chg > 0 ? "+" : ""}${pm.bp_chg ?? "—"} 口）｜歷史：成立後開高的日子收盤低於開盤 ${P(ps.B.fade)}（${ps.B.n} 天）</span></li>
    </ul>
    <p class="muted">盤前規則驗證（${ps.period[0]}～${ps.period[1]}，${ps.gap_days} 個開高 ≥${st.gap}% 的日子）：所有開高日只有 ${P(ps.base.fade)} 收盤低於開盤、${P(ps.base.flat)} 回到平盤附近，開盤到收盤平均 ${ps.base.oc > 0 ? "+" : ""}${ps.base.oc}%。這兩條盤前規則在這段期間<b>沒有明顯提高</b>開高走低的機率，請當作參考。</p>`;
  }
  if (it && it.checks) {
    const rows = Object.entries(it.checks).map(([t, c]) => `<tr><td>${t}</td><td class="num">${c.n_below}</td><td class="num"><b>${P(c.fade_below)}</b></td><td class="num">${P(c.flat_below)}</td><td class="num">${P(c.fade_above)}</td></tr>`).join("");
    html += `<details open><summary>盤中確認：開高後跌破開盤價（${it.period ? it.period[0] + "～" + it.period[1] : ""}，${it.days} 個開高日）</summary>
      <div class="scroll"><table><tr><th>到這個時間</th><th>已跌破開盤價的天數</th><th>收盤低於開盤</th><th>回到平盤附近</th><th>沒跌破時收低於開</th></tr>${rows}</table></div>
      <p class="muted">盤中每分鐘檢查：加權開高 ≥${st.gap}% 後跌破開盤價，就發「開高走低警示」，盤中進場通知也會加註保守進場。</p></details>`;
  }
  $("#gapfade").innerHTML = html;
}
loaders.market = async () => {
  loadForecast().then(() => loadGapfade(window.__fc)).catch(() => {});
  const m = await api("/api/market").catch((e) => ({ errors: [e.message] }));
  const idx = m.indices || {};
  const tiles = Object.values(idx).map((x) => `<div class="tile"><div class="k">${x.name}　${esc(x.time || "")}</div>
     <div class="v ${x.change_pct > 0 ? "up" : x.change_pct < 0 ? "down" : ""}">${fmt(x.price)}</div><div>${pct(x.change_pct)}　高 ${fmt(x.high)}／低 ${fmt(x.low)}</div></div>`);
  const v = m.verdict || {};
  tiles.push(`<div class="tile"><div class="k">今日盤勢判斷</div><div class="v stance ${v.stance === "進攻" ? "up" : v.stance === "防守" ? "down" : ""}">${v.stance || "—"}</div><div class="muted">綜合分數 ${v.score ?? "—"}</div></div>`);
  $("#idx").innerHTML = tiles.join("");
  $("#verdict").innerHTML = `<h3>大盤判斷依據</h3><ul class="list">${(v.signals || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
     <p class="muted">${esc(v.note || "")}</p>${(m.errors || []).length ? `<p class="muted">部分資料暫時無法取得：${esc(m.errors.join("；"))}</p>` : ""}`;
  const s = m.spot, f = m.futures, o = m.options, fx = m.fx, b = m.basis;
  $("#chips-mk").innerHTML = `<table>
    <tr><th>三大法人現貨（${s?.date || "—"}，億元）</th><td class="num">外資 ${signed(s?.foreign, 1)}　投信 ${signed(s?.trust, 1)}　自營 ${signed(s?.dealer, 1)}</td></tr>
    <tr><th>外資台指期淨未平倉（口）</th><td class="num">${signed(f?.net_oi)}（${signed(f?.change)}）</td></tr>
    <tr><th>外資選擇權 BC／BP 未平倉變化</th><td class="num">BC ${signed(o?.bc_change)}　BP ${signed(o?.bp_change)}</td></tr>
    <tr><th>期現貨價差（近月 − 加權）</th><td class="num">${b?.basis ?? "—"}（前日 ${b?.prev ?? "—"}）</td></tr>
    <tr><th>美元／台幣</th><td class="num">${fx?.usd_twd ?? "—"}（${signed(fx?.change, 3)}）</td></tr></table>`;
  $("#sectors").innerHTML = `<table><tr><th>族群</th><th class="num">漲跌</th><th class="num">5 日</th><th class="num">資金占比</th><th class="num">上漲家數</th><th>龍頭</th></tr>
    ${(m.sectors || []).map((x) => `<tr><td>${esc(x.industry)}</td><td class="num">${pct(x.change_pct)}</td><td class="num">${pct(x.change_5d)}</td><td class="num">${x.value_share}%</td><td class="num">${x.breadth}%</td><td><a href="#" data-sid="${x.leader}">${x.leader}</a></td></tr>`).join("")}</table>`;
  $("#topvalue").innerHTML = `<table><tr><th>代號</th><th>名稱</th><th>族群</th><th class="num">收盤</th><th class="num">漲跌</th><th class="num">成交值(億)</th></tr>
    ${(m.top_value || []).map((x) => `<tr><td><a href="#" data-sid="${x.stock_id}">${x.stock_id}</a></td><td>${esc(x.name)}</td><td>${esc(x.industry)}</td><td class="num">${fmt(x.close)}</td><td class="num">${pct(x.change_pct)}</td><td class="num">${fmt(x.value_e, 1)}</td></tr>`).join("")}</table>`;
};

document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-sid]");
  if (a) { e.preventDefault(); openStock(a.dataset.sid); }
});

// ---------------- 推薦 ----------------
function planHtml(p) {
  const sz = p.sizing || {};
  const tg = (p.targets || []).map((t) => `${fmt(t.price)}<span class="muted">（${esc(t.label)}）</span>`).join("<br>") || "—";
  return `<dl class="plan">
    <dt>現價</dt><dd>${fmt(p.price)}</dd>
    <dt>進場區</dt><dd><b>${fmt(p.entry_low)} ~ ${fmt(p.entry_high)}</b>　<span class="muted">可承受虧損內最高 ${fmt(sz.max_entry)}</span></dd>
    <dt>停損</dt><dd class="down"><b>${fmt(p.stop)}</b>　<span class="muted">${esc(p.stop_reason)}</span></dd>
    <dt>目標</dt><dd>${tg}</dd>
    ${p.healthy_pullback ? `<dt>健康回檔價</dt><dd>${fmt(p.healthy_pullback)}</dd>` : ""}
    ${p.reward_risk ? `<dt>報酬／風險</dt><dd>${p.reward_risk}</dd>` : ""}
    <dt>建議部位</dt><dd>${fmt(sz.shares, 0)} 股（約 ${fmt(sz.cost, 0)} 元），打到停損約虧 ${fmt(sz.loss_at_stop, 0)} 元（${sz.stop_pct}%）</dd>
  </dl>`;
}
function marginHtml(m) {
  if (!m) return "";
  const head = m.recommended ? `<span class="chip up">建議可用融資</span>` : `<span class="chip">建議用現股</span>`;
  return `<details ${m.recommended ? "open" : ""}><summary>槓桿（融資）方案　${head}</summary>
    ${m.ok ? `<dl class="plan">
      <dt>融資股數</dt><dd>${fmt(m.shares, 0)} 股（${fmt(m.lots, 0)} 張）</dd>
      <dt>自備款／融資</dt><dd>${fmt(m.self_funded, 0)} ／ ${fmt(m.loan, 0)} 元（成數 ${Math.round(m.ratio * 100)}%，槓桿 ${m.leverage} 倍）</dd>
      <dt>追繳價</dt><dd class="down">${fmt(m.call_price)}（維持率跌破 130%）</dd>
      <dt>打到停損</dt><dd>約虧 ${fmt(m.loss_at_stop, 0)} 元（含預估 20 日利息 ${fmt(m.interest_est, 0)} 元）</dd></dl>` : ""}
    ${m.why_not.length ? `<ul class="list">${m.why_not.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    <p class="muted">${esc(m.note)}</p></details>`;
}
function exitRulesHtml(rules) {
  const act = { sell_all: "出場", sell_half: "減碼 50%", watch_pullback: "停利觀察", notify: "提醒", target_hit: "停利" };
  return `<ul class="list">${(rules || []).map((r) => `<li><span class="chip">${act[r.action] || r.action}</span>${esc(r.label)}${r.level ? `（${fmt(r.level)}）` : ""}</li>`).join("")}</ul>`;
}
function recCard(r) {
  const p = r.plan;
  const fund = p.fundamental;
  return `<div class="card rec" data-recid="${r.id}">
    <div class="head"><div class="title"><a href="#" data-sid="${r.stock_id}">${r.stock_id} ${esc(r.name)}</a></div><div class="score">分數 ${fmt(r.score, 1)}｜${r.trade_date}</div></div>
    <div><span class="chip">${esc(p.setup_name)}</span>${(p.tags || []).map((t) => `<span class="chip ${t === "盤中" ? "up" : ""}">${t === "盤中" ? "⚡ 盤中訊號" : esc(t)}</span>`).join("")}<span class="chip">${esc(p.industry || "")}</span></div>
    ${planHtml(p)}
    ${marginHtml(p.margin)}
    <details open><summary>為什麼推薦</summary><ul class="list">${p.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>
    ${p.warnings.length ? `<details><summary class="warn">注意事項（${p.warnings.length}）</summary><ul class="list">${p.warnings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>` : ""}
    <details><summary>停利／停損／減碼規則</summary>${exitRulesHtml(p.exit_rules)}</details>
    ${fund ? `<details><summary>基本面 ${fund.grade}（${fund.score}）</summary>${fund.highlights.map(checkHtml).join("")}</details>` : ""}
    ${p.chips ? `<details><summary>法人籌碼</summary><p class="muted">5 日：外資 ${signed(p.chips.summary.foreign["5d"])} 張、投信 ${signed(p.chips.summary.trust["5d"])} 張</p><ul class="list">${p.chips.notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>` : ""}
    <div class="btns">${r.status === "active" ? `<button data-buy="${r.id}">我已買進</button><button class="ghost" data-dismiss="${r.id}">不買</button>` : `<span class="muted">狀態：${(REC_STATUS[r.status] || [, r.status])[1]}</span>`}</div>
  </div>`;
}
let recCache = [];
async function loadIntradayWatch() {
  const w = await api("/api/intraday-watch");
  const items = w.items || [];
  $("#iw-meta").textContent = `${Object.keys(w.pre || {}).length ? `🔔 ${Object.keys(w.pre).length} 檔預備中｜` : ""}${items.length} 檔${w.date ? `｜依 ${w.date} 收盤建立` : ""}｜${w.in_session ? `盤中監控中（每 ${w.interval_sec} 秒）` : "收盤後，下一個交易日 9:00 開始監控"}`;
  const pre = w.pre || {};
  const preTxt = (sid) => (pre[sid] || []).map((p) => p.kind === "time"
    ? `🔔 ${esc(p.name)}：條件到位，${p.gate} 後維持即成立（停損 ${p.stop}）`
    : `🔔 ${esc(p.name)}：價格到 ${p.trigger}（${p.dist > 0 ? "+" : ""}${p.dist}%）即成立（停損 ${p.stop}）`).join("<br>");
  const sorted = [...items].sort((a, b) => (pre[b.stock_id] ? 1 : 0) - (pre[a.stock_id] ? 1 : 0));
  $("#iw-table").innerHTML = `<tr><th>股票</th><th>最接近的型態（收盤時）</th><th>已符合</th><th>還差</th><th>盤中預備訊號</th></tr>` + sorted.map((x) =>
    `<tr><td><a href="#" data-sid="${x.stock_id}">${x.stock_id} ${esc(x.name || "")}</a></td><td>${esc(x.setup)}</td><td class="num">${x.met}／${x.total}</td>
     <td>${x.missing.length ? x.missing.map(esc).join("、") : `<span class="muted">條件都到位，等盤中價格進入可操作範圍</span>`}</td>
     <td>${preTxt(x.stock_id) || `<span class="muted">—</span>`}</td></tr>`).join("");
}
loaders.recs = async () => {
  loadIntradayWatch().catch(() => {});
  recCache = await api(`/api/recommendations?status=${$("#rec-filter").value}`);
  $("#recs").innerHTML = recCache.map(recCard).join("") || `<p class="muted">目前沒有符合兩份講義條件的標的。系統會在收盤後（14:40、21:30）與盤中自動掃描，符合就推播通知。</p>`;
};
$("#rec-filter").addEventListener("change", loaders.recs);
$("#run-scan").addEventListener("click", async () => { await api("/api/jobs/scan", { method: "POST" }); toast("已開始掃描，約需數分鐘（基本面有 API 額度限制）"); });
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-buy]");
  if (b) {
    const r = recCache.find((x) => x.id == b.dataset.buy);
    buyDialog({ rec_id: r.id, stock_id: r.stock_id, name: r.name, price: r.plan.price, shares: r.plan.sizing.shares, stop: r.plan.stop, stop_reason: r.plan.stop_reason, margin: r.plan.margin });
  }
  const d = e.target.closest("[data-dismiss]");
  if (d) { await api(`/api/recommendations/${d.dataset.dismiss}/dismiss`, { method: "POST" }); loaders.recs(); }
});

// ---------------- 對話框 ----------------
function dialog(html, onSubmit) {
  const dlg = $("#dlg"), f = $("#dlg-form");
  f.innerHTML = html + `<div class="btns"><button value="ok">確認</button><button class="ghost" value="cancel" formnovalidate>取消</button></div>`;
  f.onsubmit = async (e) => {
    if (e.submitter?.value !== "ok") return;
    e.preventDefault();
    try { await onSubmit(Object.fromEntries(new FormData(f))); dlg.close(); } catch (err) { toast("失敗：" + err.message); }
  };
  dlg.showModal();
}
function buyDialog(d) {
  dialog(`<h3>確認買進 ${esc(d.stock_id)} ${esc(d.name || "")}</h3>
    ${d.rec_id ? "" : `<label>股票代號<input name="stock_id" required value="${esc(d.stock_id || "")}"></label>`}
    <label>交易方式<select name="mode" id="buy-mode">
      <option value="cash">現股</option>
      <option value="margin" ${d.margin?.recommended ? "selected" : ""}>融資（槓桿）${d.margin ? (d.margin.recommended ? "｜系統建議" : "｜系統不建議") : ""}</option></select></label>
    <p class="muted" id="mode-hint"></p>
    <label>成交價<input name="price" type="number" step="0.01" required value="${d.price ?? ""}"></label>
    <label>股數（1 張 = 1000 股；融資須整張）<input name="shares" id="buy-shares" type="number" step="1" required value="${d.margin?.recommended ? d.margin.shares : (d.shares ?? 1000)}"></label>
    <label>買進日期<input name="buy_date" type="date" value="${new Date().toISOString().slice(0, 10)}"></label>
    <label>券商帳上的總成本（含手續費，選填）<input name="cost" type="number" step="any" placeholder="不填則以成交價＋手續費計算"></label>
    <p>停損：${d.stop ? `<b class="down">${fmt(d.stop)}</b>　<span class="muted">${esc(d.stop_reason || "")}</span>` : `<span class="muted">由系統依目前型態計算</span>`}</p>
    <p class="muted">停損位置看的是「進場要素消失」的價位，跟你的成交價無關，由系統決定，之後收盤會依規則自動上移（只上不下）。確認後開始盤中監控，該停損、減碼、停利時推播 LINE。</p>`,
    async (v) => {
      const r = await api("/api/positions", { method: "POST", body: { rec_id: d.rec_id || null, stock_id: v.stock_id || d.stock_id, price: +v.price, shares: +v.shares, buy_date: v.buy_date, mode: v.mode, cost: v.cost ? +v.cost : null } });
      loaders.recs(); loaders.portfolio();
      if (r.warnings?.length) setTimeout(() => dialog(`<h3>已記錄買進，但請注意</h3><ul class="list">${r.warnings.map((w) => `<li class="warn">${esc(w)}</li>`).join("")}</ul><p>系統停損：<b class="down">${fmt(r.stop)}</b>　<span class="muted">${esc(r.stop_reason || "")}</span></p>`, async () => {}), 300);
      else toast(`已記錄買進，停損 ${fmt(r.stop)}，開始監控`);
    });
  const sel = $("#buy-mode"), hint = $("#mode-hint"), sh = $("#buy-shares");
  const upd = () => {
    const m = d.margin;
    if (sel.value === "margin") {
      hint.innerHTML = m ? (m.ok ? `建議 ${fmt(m.shares, 0)} 股，自備款約 ${fmt(m.self_funded, 0)} 元，追繳價 ${fmt(m.call_price)}${m.why_not.length ? `<br><span class="warn">${esc(m.why_not.join("；"))}</span>` : ""}` : `<span class="warn">${esc(m.reason)}</span>`) : "融資成數依設定頁（上市 6 成、上櫃 5 成），系統會計算追繳價並監控維持率";
      if (m?.ok) sh.value = m.shares;
    } else { hint.textContent = ""; if (d.shares) sh.value = d.shares; }
  };
  sel.addEventListener("change", upd); upd();
}
$("#add-pos").addEventListener("click", () => buyDialog({}));
$("#reset-portfolio").addEventListener("click", async () => {
  const h = await api("/api/portfolio/health");
  const n = h.positions.length;
  dialog(`<h3>重新設定持股健檢</h3>
    <p>會清空<b>全部持股（目前 ${n} 筆）與交易紀錄</b>，已實現損益、勝率也會歸零，盤中監控停止追蹤這些持股。</p>
    <p class="muted">不影響推薦、通知與設定。清空前會自動備份資料庫，清空後可以按「復原清空前的持股」改回來。</p>`,
    async () => {
      const r = await api("/api/portfolio/reset", { method: "POST" });
      $("#undo-portfolio").classList.toggle("hidden", !r.can_undo);
      toast(r.cleared || r.trades ? `已清空 ${r.cleared} 筆持股、${r.trades} 筆交易紀錄` : "目前沒有持股資料");
      loaders.portfolio();
    });
});
$("#undo-portfolio").addEventListener("click", async () => {
  const r = await api("/api/portfolio/undo-reset", { method: "POST" });
  $("#undo-portfolio").classList.add("hidden");
  toast(`已復原 ${r.restored} 筆持股、${r.trades} 筆交易紀錄`); loaders.portfolio();
});

// ---------------- 持股健檢 ----------------
const STATUS = { ok: "健康", warn: "注意", danger: "危險" };
loaders.portfolio = async () => {
  const h = await api("/api/portfolio/health");
  const s = h.summary;
  $("#pf-summary").innerHTML = [
    ["持股市值", fmt(s.market_value, 0)], ["自有資金使用率", s.exposure_pct == null ? "—" : s.exposure_pct + "%"],
    ["未實現損益", signed(s.unrealized_pnl)], ["全部打到停損的虧損", fmt(s.risk_at_stop, 0)],
    ["融資借款", fmt(s.margin_loan, 0)], ["已實現損益", signed(s.realized_pnl)], ["勝率", s.win_rate == null ? "—" : `${s.win_rate}%（${s.closed_trades} 筆）`],
  ].map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#pf-notes").innerHTML = h.notes.length ? `<div class="card"><h3>投資組合提醒</h3><ul class="list">${h.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>` : "";
  $("#positions").innerHTML = h.positions.map((p) => `<div class="card status-${p.status}">
     <div class="rec head"><div class="title"><a href="#" data-sid="${p.stock_id}">${p.stock_id} ${esc(p.name)}</a> <span class="chip">${STATUS[p.status]}</span></div>
       <div class="score">${esc(p.setup)}｜${p.buy_date}</div></div>
     <dl class="plan">
       <dt>成本</dt><dd>${fmt(p.cost, 0)} 元（成本均價 ${fmt(p.avg_cost, 2)}）</dd>
       <dt>現價／市值</dt><dd><b>${fmt(p.last)}</b>　市值 ${fmt(p.value, 0)}</dd>
       <dt>未實現損益</dt><dd><b>${signed(p.pnl)} 元　${pct(p.pnl_pct)}</b><br><span class="muted">已扣預估賣出手續費 ${fmt(p.sell_fee, 0)}、證交稅 ${fmt(p.sell_tax, 0)}</span></dd>
       <dt>損益平衡點</dt><dd><b>${fmt(p.break_even)}</b>　<span class="muted">（精確 ${fmt(p.break_even_exact, 4)}，${p.to_break_even_pct > 0 ? `還差 ${p.to_break_even_pct}%` : "已在平衡點之上"}）</span></dd>
       <dt>持股</dt><dd>${fmt(p.shares, 0)} 股　<span class="chip">${p.mode === "margin" ? "融資" : "現股"}</span></dd>
       ${p.margin ? `<dt>融資</dt><dd>自備 ${fmt(p.margin.self_funded, 0)}／借 ${fmt(p.margin.loan, 0)} 元，已計息 ${p.margin.days} 天約 ${fmt(p.margin.interest, 0)} 元（損益已扣）</dd>
       <dt>維持率</dt><dd class="${p.margin.maintenance_ratio < 150 ? "danger" : ""}"><b>${p.margin.maintenance_ratio}%</b>　追繳價 ${fmt(p.margin.call_price)}　自有資金報酬 ${pct(p.margin.return_on_equity)}</dd>` : ""}
       <dt>停損（系統）</dt><dd><b class="down">${fmt(p.stop)}</b>（距離 ${p.to_stop_pct ?? "—"}%）<br><span class="muted">${esc(p.stop_reason || "")}</span></dd>
       <dt>下一個目標</dt><dd>${p.next_target ? `${fmt(p.next_target.price)}（${esc(p.next_target.label)}）` : "—"}</dd>
       <dt>R 倍數</dt><dd>${p.r_multiple ?? "—"}</dd>
       <dt>技術面</dt><dd>${esc(p.context.level || "")}・${esc(p.context.regime || "")}・格局 ${esc(p.context.structure || "")}　基本面 ${p.fundamental_grade || "—"}</dd>
     </dl>
     ${p.notes.length ? `<ul class="list">${p.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : `<p class="muted">進場要素仍在，依計畫續抱。</p>`}
     ${(p.entry_warnings || []).map((w) => `<p class="warn">⚠️ ${esc(w)}</p>`).join("")}
     <details><summary>出場規則</summary>${exitRulesHtml(p.exit_rules)}</details>
     ${(p.stop_history || []).length > 1 ? `<details><summary>停損移動紀錄</summary><ul class="list">${p.stop_history.map((h) => `<li>${h.date}：${fmt(h.stop)}（${esc(h.reason)}）</li>`).join("")}</ul></details>` : ""}
     <div class="btns"><button class="small" data-sell="${p.id}" data-sh="${p.shares}" data-px="${p.last}">記錄賣出／減碼</button><button class="small ghost" data-cost="${p.id}" data-cur="${p.cost}">對帳：修改成本</button></div>
   </div>`).join("") || `<p class="muted">尚無持股。到「推薦」按「我已買進」，或用上方按鈕新增。</p>`;
  const t = await api("/api/trades");
  $("#trades").innerHTML = `<table><tr><th>時間</th><th>股票</th><th>買賣</th><th class="num">價格</th><th class="num">股數</th><th>原因</th></tr>
    ${t.map((x) => `<tr><td>${x.ts.replace("T", " ")}</td><td>${x.stock_id} ${esc(x.name)}</td><td class="${x.side === "buy" ? "up" : "down"}">${x.side === "buy" ? "買進" : "賣出"}</td><td class="num">${fmt(x.price)}</td><td class="num">${fmt(x.shares, 0)}</td><td>${esc(x.reason)}</td></tr>`).join("")}</table>`;
};
document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-cost]");
  if (c) dialog(`<h3>對帳：修改成本</h3><label>券商帳上的總成本（含買進手續費）<input name="cost" type="number" step="any" required value="${c.dataset.cur}"></label>
      <p class="muted">以元大為例：庫存明細的「成本」或「付出成本」。系統會依此重算未實現損益與損益平衡點。</p>`,
    async (v) => { await api(`/api/positions/${c.dataset.cost}/cost`, { method: "POST", body: { cost: +v.cost } }); toast("已更新成本"); loaders.portfolio(); });
  const s = e.target.closest("[data-sell]");
  if (s) dialog(`<h3>記錄賣出</h3><label>成交價<input name="price" type="number" step="0.01" required value="${s.dataset.px}"></label>
      <label>股數<input name="shares" type="number" required value="${s.dataset.sh}"></label>
      <label>原因<select name="reason"><option>停損：進場要素消失</option><option>減碼：跌破季線／五日線</option><option>停利：目標價達成</option><option>手動賣出</option></select></label>`,
    async (v) => { const r = await api(`/api/positions/${s.dataset.sell}/sell`, { method: "POST", body: { price: +v.price, shares: +v.shares, reason: v.reason } }); toast(`已記錄，損益 ${r.pnl} 元`); loaders.portfolio(); });
});

// ---------------- 個股 ----------------
let chart;
function checkHtml(c) {
  const icon = { good: "✅", warn: "⚠️", bad: "⛔", info: "・" }[c.status] || "・";
  return `<div class="check"><span>${icon}</span><span><b>${esc(c.name)}</b>　<span class="muted">${esc(c.detail)}</span></span></div>`;
}
async function openStock(sid) {
  show("stock");
  $("#q").value = sid;
  $("#stock-body").innerHTML = `<p class="muted">載入 ${sid}…（第一次查詢需抓財報，約 10 秒）</p>`;
  let d;
  try { d = await api(`/api/stock/${encodeURIComponent(sid)}`); } catch (e) { $("#stock-body").innerHTML = `<p class="danger">${esc(e.message)}</p>`; return; }
  sid = d.info.stock_id;  // 輸入名稱時，改用實際代號
  $("#q").value = sid;
  const q = d.quote, t = d.technical, f = d.fundamental, c = d.chips, ctx = t?.context || {};
  const isEtf = sid.startsWith("00");
  const sigs = (t?.signals || []).sort((a, b) => b.triggered - a.triggered || b.score - a.score);
  $("#stock-body").innerHTML = `
    <div class="grid tiles">
      <div class="tile"><div class="k">${esc(d.info.name)}（${d.info.market === "tpex" ? "上櫃" : "上市"}・${esc(d.info.industry)}）</div>
        <div class="v">${fmt(q?.price ?? ctx.close)}</div><div>${q ? pct(q.prev_close ? (q.price / q.prev_close - 1) * 100 : null) : ""}　${esc(q?.time || "")}</div></div>
      <div class="tile"><div class="k">位階／角度</div><div class="v" style="font-size:18px">${esc(ctx.level || "—")}</div><div>${esc(ctx.regime || "")} ${ctx.angle ?? ""}°</div></div>
      <div class="tile"><div class="k">格局</div><div class="v" style="font-size:18px">${{ uptrend: "高過高、低不過低", downtrend: "低過低、高不過高", turning_up: "高過高（可能轉多）", turning_down: "低過低（可能做頭）", range: "震盪" }[ctx.structure] || "—"}</div></div>
      <div class="tile"><div class="k">量比</div><div class="v">${ctx.volume_ratio ?? "—"}</div><div class="muted">>1 代表今日量較大</div></div>
      <div class="tile"><div class="k">基本面</div><div class="v">${isEtf ? "不適用" : f ? `${f.grade}（${f.score}）` : "—"}</div><div class="muted">${isEtf ? "ETF 不做公司健檢" : f ? (f.pass ? "體質合格" : "未通過健檢") : esc(d.fundamental_error || "資料更新中")}</div></div>
    </div>
    ${verdictHtml(d.verdict, sid)}
    <div class="card"><div id="chart"></div><p class="muted">K 線：紅漲綠跌；均線 5／20／60／240。${sigs.some((s) => s.triggered) ? "虛線為已成立型態的停損（綠）與目標（紅）。" : "目前沒有成立的進場型態，因此不畫停損與目標線。"}</p></div>
    <div class="grid two">
      <div class="card"><h3>進場型態檢查</h3>
        ${sigs.map((s) => `<details ${s.triggered ? "open" : ""}><summary>${s.triggered ? "🟢 符合" : "⚪ 未符合"}　${esc(s.name)}</summary>
          <ul class="list">${s.reasons.map((x) => `<li>✓ ${esc(x)}</li>`).join("")}${s.warnings.map((x) => `<li class="warn">! ${esc(x)}</li>`).join("")}</ul>
          ${s.triggered && s.stop ? `<p class="muted">停損 ${fmt(s.stop)}（${esc(s.stop_reason)}）　目標 ${(s.targets || []).map((x) => fmt(x.price)).join("、") || "—"}</p>` : `<p class="muted">型態未成立，沒有進場要素，因此不提供停損與目標價。</p>`}
          ${s.triggered ? `<button class="small" data-manualbuy="${sid}" data-px="${q?.price ?? ctx.close}" data-stop="${s.stop}" data-reason="${esc(s.stop_reason)}">我已買進</button>` : ""}</details>`).join("")}
        <p class="muted">${ctx.wave ? `波段目標價：起漲 ${fmt(ctx.wave.rally_start)} → 壓力 ${fmt(ctx.wave.resistance)}，目標 ${fmt(ctx.wave.target)}，健康回檔 ${fmt(ctx.wave.healthy_pullback)}` : "波段目標價：近期沒有「急漲段＋橫盤整理」可套用 1:1 等幅"}</p>
      </div>
      <div class="card"><h3>基本面健檢（從 0 到 1 選出好公司）</h3>${isEtf ? "<p class='muted'>ETF 不做公司基本面健檢</p>" : f ? f.checks.map(checkHtml).join("") : "<p class='muted'>基本面資料尚未取得（全市場更新中）</p>"}
        ${f?.metrics ? `<p class="muted">本益比 ${f.metrics.per ?? "—"}・股價淨值比 ${f.metrics.pbr ?? "—"}・殖利率 ${f.metrics.dividend_yield ?? "—"}%</p>` : ""}</div>
    </div>
    <div class="card"><h3>法人籌碼（張）</h3>${c ? `<p>5 日：外資 ${signed(c.summary.foreign["5d"])}、投信 ${signed(c.summary.trust["5d"])}、自營 ${signed(c.summary.dealer["5d"])}　｜　20 日：外資 ${signed(c.summary.foreign["20d"])}、投信 ${signed(c.summary.trust["20d"])}</p>
      <ul class="list">${c.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : `<p class="muted">暫無法人資料${d.chips_error ? "（資料來源額度暫時用完，稍後再查）" : ""}</p>`}</div>`;
  drawChart(d.bars, sigs.find((s) => s.triggered));
  const vb = $("#verdict-buy");
  if (vb) vb.addEventListener("click", () => {
    const p = d.verdict.plan;
    buyDialog({ rec_id: d.verdict.rec_id, stock_id: sid, name: d.info.name, price: p.price, shares: p.sizing.shares, stop: p.stop, stop_reason: p.stop_reason, margin: p.margin });
  });
}
function expBar(exp, p) {
  // 期望報酬刻度：−3% ～ +4%
  const lo = -3, hi = 4, pos = (x) => Math.max(0, Math.min(100, (x - lo) / (hi - lo) * 100));
  const col = exp >= p.buy_exp ? "var(--up)" : exp <= p.avoid_exp ? "var(--muted)" : "var(--warn)";
  return `<div class="ratebar"><span class="fill" style="width:${pos(exp)}%;background:${col}"></span>
    <span class="mark" style="left:${pos(p.avoid_exp)}%"></span><span class="mark" style="left:${pos(p.buy_exp)}%"></span></div>
    <div class="ratebar-legend"><span>${lo}%</span><span style="left:${pos(p.avoid_exp)}%">不推 ≤ ${p.avoid_exp}%</span><span style="left:${pos(p.buy_exp)}%">推薦 ≥ +${p.buy_exp}%</span><span>+${hi}%</span></div>`;
}
function winrateHtml(w) {
  if (!w) return `<p class="muted">勝率模型尚未訓練（到「勝率」頁按重新訓練）。</p>`;
  const p = w.params;
  return `<div class="wr">
    <div class="wr-head"><span class="wr-rate ${w.exp > 0 ? "up" : "down"}">${w.exp > 0 ? "+" : ""}${w.exp}%</span><span>每筆期望報酬　<span class="muted">勝率 ${Math.round(w.rate * 100)}%</span></span></div>
    ${expBar(w.exp, p)}
    <p class="muted">${w.decision === "buy" ? "依據（已通過驗證的推薦組合）" : w.n_rules > 1 ? `綜合 ${w.n_rules} 個符合的條件組合（依樣本數加權）；代表組合` : "依據"}：${w.rule ? esc(w.rule.names.join(" + ")) + `（${w.rule.n.toLocaleString()} 筆；訓練 期望 ${w.rule.train_exp > 0 ? "+" : ""}${w.rule.train_exp}%・勝率 ${Math.round(w.rule.train_rate * 100)}%／驗證 期望 ${w.rule.test_exp > 0 ? "+" : ""}${w.rule.test_exp}%・勝率 ${Math.round(w.rule.test_rate * 100)}%）` : `沒有符合的條件組合，以全市場平均計`}</p>
    <p class="muted">交易定義：隔日開盤買進；${esc(p.exit_desc || "")}（已扣手續費與稅）。推薦需兩期都 勝率 ≥ ${Math.round(p.min_rate * 100)}%、期望 > +${p.buy_exp}%、虧損超過 6% 的比例 ≤ ${((p.max_big ?? 0.02) * 100).toFixed(0)}%。</p>
    ${w.matched.length > 1 ? `<details><summary>其他符合的條件組合</summary><ul class="list">${w.matched.map((r) => `<li>${esc(r.names.join(" + "))}：期望 ${r.exp > 0 ? "+" : ""}${r.exp}%、勝率 ${Math.round(r.rate * 100)}%（${r.n} 筆）</li>`).join("")}</ul></details>` : ""}
  </div>`;
}
function vSection(title, body) {
  return `<div class="v-sec"><div class="v-sec-title">${title}</div>${body}</div>`;
}
function vList(items, cls = "") {
  return items.length ? `<ul class="list">${items.map((x) => `<li class="${cls}">${cls === "warn" ? "! " : ""}${esc(x)}</li>`).join("")}</ul>` : `<p class="muted">無</p>`;
}
function nearHtml(nm) {
  if (!nm) return `<p class="muted">勝率模型尚未訓練（勝率頁按「重新訓練」），或此股資料不足</p>`;
  if (!nm.items.length) return `<p class="muted">沒有任何進場型態達到多數條件（至少三分之二）</p>`;
  const pct = (x) => (x * 100).toFixed(0) + "%";
  const tag = { buy: ["up", "達推薦門檻"], avoid: ["down", "歷史虧損"], neutral: ["muted", "未達門檻"], unknown: ["muted", "樣本不足"] };
  return nm.items.map((it) => {
    const st = it.stats, [c, t] = tag[it.decision];
    const nums = st ? `勝率 <b>${pct(st.rate)}</b>（訓練 ${pct(st.train_rate)}／驗證 ${pct(st.test_rate)}）・每筆期望 <b class="${st.exp >= 0 ? "up" : "down"}">${st.exp >= 0 ? "+" : ""}${st.exp.toFixed(2)}%</b>・${st.n} 筆`
      : "歷史上同樣情況的樣本太少，無法估計";
    return `<div class="near"><div><b>${esc(it.name)}</b> <span class="muted">符合 ${it.met.length}／${it.total}</span> <span class="${c}">【${t}】</span></div>
      <div>${nums}</div>
      ${it.missing.length ? `<div class="muted">未成立：${it.missing.map(esc).join("、")}</div>` : ""}
      ${it.basis ? `<div class="muted">比對依據：${esc(it.basis)}</div>` : ""}</div>`;
  }).join("") + `<p class="muted">一個條件沒到位不直接否決：回測過去一年「同型態、缺同樣條件」的所有交易，訓練期與驗證期都達推薦門檻（期望報酬、勝率）才建議買進。</p>`;
}
function basisHtml(v, isEtf) {
  const b = v.basis;
  if (!b) return `<p class="muted">${isEtf ? "ETF 不在勝率模型範圍（模型只用個股訓練），只依講義規則判斷" : "勝率模型尚未訓練，或此股資料不足以比對；結論只依講義規則"}</p>`;
  const p = b.params, sign = (x) => (x > 0 ? "+" : "") + x;
  const gate = `推薦門檻：訓練、驗證兩期都要 勝率 ≥ ${Math.round((p.min_rate ?? p.gate?.min_rate ?? 0.6) * 100)}%、每筆期望 > +${p.buy_exp ?? p.min_exp}%、虧損超過 6% 的比例 ≤ ${((p.max_big ?? 0.02) * 100).toFixed(0)}%`;
  let verdictLine;
  if (v.decision === "buy" && b.pass) verdictLine = `<span class="up">✅ 達推薦門檻 → 建議買進</span>`;
  else if (v.decision === "buy") verdictLine = `<span class="up">✅ 講義進場型態全部成立且通過風控 → 建議買進</span>（歷史數字未達門檻，僅供參考）`;
  else if (b.pass) verdictLine = `<span class="warn">歷史數字達標，但${esc(v.summary || "其他條件未通過")} → ${esc(v.label)}</span>`;
  else verdictLine = `<span class="${v.decision === "avoid" ? "down" : "muted"}">未達推薦門檻 → ${esc(v.label)}</span>`;
  const split = b.train_rate != null ? `（訓練 期望 ${sign(b.train_exp)}%・勝率 ${Math.round(b.train_rate * 100)}%／驗證 期望 ${sign(b.test_exp)}%・勝率 ${Math.round(b.test_rate * 100)}%）` : "";
  return `<div class="wr">
    <div class="wr-head"><span class="wr-rate ${b.exp > 0 ? "up" : "down"}">${sign(b.exp)}%</span><span>每筆期望報酬　<span class="muted">勝率 ${Math.round(b.rate * 100)}%${b.n ? `・${b.n.toLocaleString()} 筆` : ""}</span></span></div>
    ${expBar(b.exp, p)}
    <p><b>${verdictLine}</b></p>
    <p class="muted">依據：${esc(b.title)}${b.desc ? `（${esc(b.desc)}）` : ""}${split}</p>
    <p class="muted">${gate}。交易定義：${b.source === "ml" && v.ml ? `隔日開盤買進；${esc(v.ml.params.desc || "")}（排除鎖漲停買不到、鎖跌停賣不掉；已扣手續費與稅）` : `隔日開盤買進；${esc(p.exit_desc || "")}（跳空以開盤價成交、排除鎖漲停買不到與鎖跌停賣不掉；已扣手續費與稅）`}。</p>
  </div>`;
}
const pctS = (x) => (x == null ? "—" : (x >= 0 ? "+" : "") + x.toFixed(2) + "%");
function mlHtml(ml) {
  if (!ml) return `<p class="muted">AI 模型尚未訓練（勝率頁 →「重新訓練 AI 模型」），或此股不在模型範圍</p>`;
  const top = ml.rank == null ? null : Math.max(1, Math.round((1 - ml.rank) * 100));
  const mt = ml.market_trend === 1 ? `<span class="up">多頭（加權指數在 20MA、60MA 之上）</span>` : ml.market_trend === 0 ? `<span class="down">空頭體制（指數在 20MA 或 60MA 之下）→ 不發多頭訊號</span>` : "—";
  const b = ml.bucket;
  const stat = b ? `樣本外同樣排名：勝率 <b>${Math.round(b.all.rate * 100)}%</b>・每筆期望 <b class="${b.all.exp >= 0 ? "up" : "down"}">${pctS(b.all.exp)}</b>・虧損超過 6% 的比例 ${(b.all.big_loss_rate * 100).toFixed(1)}%（${b.all.n.toLocaleString()} 筆）` : "排名未進入任何門檻區間";
  const state = !ml.enabled ? `<span class="muted">模型未通過樣本外門檻，目前只顯示排名、不發推薦</span>`
    : ml.signal ? `<span class="up">✅ 模型訊號成立（排名達前 ${Math.round((1 - ml.threshold) * 100)}%、大盤多頭）</span>`
    : `<span class="muted">未達模型訊號（需排名前 ${Math.round((1 - ml.threshold) * 100)}% 且大盤多頭）</span>`;
  return `<div>成功機率 <b>${Math.round(ml.prob * 100)}%</b>${top ? `・今日主訊號中排名前 <b>${top}%</b>` : ""}</div>
    <div>${stat}</div><div>大盤體制：${mt}</div><div>${state}</div>
    <p class="muted">${ml.is_event ? "今天屬於主訊號（講義型態多數條件成立），模型判斷這次訊號成功的機率。" : "今天不是主訊號（沒有講義型態達多數條件），模型不評分。"}出場：${esc(ml.params.desc || "")}；已排除鎖漲停買不到、鎖跌停賣不掉、跳空以開盤價成交。</p>`;
}
function verdictHtml(v, sid) {
  if (!v) return "";
  const cls = { buy: "verdict-buy", wait: "verdict-wait", avoid: "verdict-avoid" }[v.decision];
  const icon = { buy: "🟢", wait: "🟡", avoid: "⛔" }[v.decision];
  const p = v.plan;
  const isEtf = sid.startsWith("00");
  // 固定六段：結論 → 期望報酬 → 判斷理由 → 進場條件 → 注意事項 → 操作計畫
  const wr = v.winrate ? winrateHtml(v.winrate)
    : `<p class="muted">${isEtf ? "ETF 不在勝率模型範圍（模型只用個股訓練）" : "勝率模型尚未訓練，或此股資料不足以比對"}</p>`;
  let conds;
  if (v.closest) {
    const total = Math.max(v.closest.met.length + v.closest.missing.length, 1);
    conds = `<div class="conds"><div class="conds-title">${esc(v.closest.name)} <span class="muted">｜已符合 ${v.closest.met.length}／${total}</span></div>
      <div class="conds-bar"><span style="width:${Math.round(v.closest.met.length / total * 100)}%"></span></div>
      ${v.closest.missing.map((x) => `<div class="cond todo"><span class="box"></span>${esc(x)}</div>`).join("")}
      ${v.closest.met.map((x) => `<div class="cond done"><span class="box">✓</span>${esc(x)}</div>`).join("")}</div>`;
  } else if ((v.triggered || []).length) {
    conds = `<div class="conds">${v.triggered.map((x) => `<div class="cond done"><span class="box">✓</span>已成立：${esc(x)}</div>`).join("")}</div>`;
  } else {
    conds = `<p class="muted">沒有任何進場型態接近成立</p>`;
  }
  const near = nearHtml(v.near);
  const plan = p && v.decision === "buy"
    ? `${planHtml(p)}${marginHtml(p.margin)}<details><summary>停利／停損／減碼規則</summary>${exitRulesHtml(p.exit_rules)}</details>
      <div class="btns"><button id="verdict-buy">我已買進</button></div>`
    : `<p class="muted">${v.decision === "avoid" ? "不建議買進，不提供進場、停損與目標價" : "尚未符合進場條件，條件成立後才提供進場區、停損與目標價"}</p>`;
  return `<div class="card verdict ${cls}">
    <div class="verdict-head"><span class="verdict-label">${icon} ${esc(v.label)}</span><span>${esc(v.summary || "—")}</span></div>
    ${vSection("結論依據（歷史期望報酬）", basisHtml(v, isEtf))}
    ${vSection("判斷理由", vList(v.reasons))}
    ${vSection("進場條件", conds)}
    ${vSection("多數條件勝率", near)}
    ${vSection("AI 模型（LightGBM）", mlHtml(v.ml))}
    ${v.winrate ? `<details class="v-sec"><summary class="muted">其他統計參考：整體條件組合模型（不影響上方結論）</summary>${wr}</details>` : ""}
    ${vSection("注意事項", vList(v.cautions, "warn"))}
    ${vSection("操作計畫", plan)}
    <p class="muted">判斷順序：歷史期望報酬（達推薦門檻直接推薦、≤ 不推門檻不推薦）→ 基本面健檢 → 位階與格局 → 進場型態（全部成立，或多數成立且歷史勝率達門檻）→ 風控 → 大盤與籌碼。僅為規則與統計判斷，不是投資建議。</p>
  </div>`;
}
function drawChart(bars, sig) {
  const el = $("#chart");
  if (!el || !window.LightweightCharts || !bars.length) return;
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const css = getComputedStyle(document.documentElement);
  chart = LightweightCharts.createChart(el, {
    autoSize: true, layout: { background: { color: "transparent" }, textColor: css.getPropertyValue("--muted") },
    grid: { vertLines: { color: dark ? "#222833" : "#f0f1f4" }, horzLines: { color: dark ? "#222833" : "#f0f1f4" } },
    rightPriceScale: { mode: LightweightCharts.PriceScaleMode.Logarithmic }, timeScale: { borderVisible: false },
  });
  const up = css.getPropertyValue("--up").trim(), down = css.getPropertyValue("--down").trim();
  const k = chart.addCandlestickSeries({ upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down });
  k.setData(bars.map((b) => ({ time: b.date, open: b.open, high: b.high, low: b.low, close: b.close })));
  const colors = { ma5: "#f2a33a", ma20: "#7c5cff", ma60: "#2aa1d8", ma240: "#999" };
  for (const [key, color] of Object.entries(colors)) {
    const s = chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    s.setData(bars.filter((b) => b[key] != null).map((b) => ({ time: b.date, value: b[key] })));
  }
  const vol = chart.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" } });
  chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  vol.setData(bars.map((b) => ({ time: b.date, value: b.volume, color: b.close >= b.open ? up + "66" : down + "66" })));
  if (sig?.stop) k.createPriceLine({ price: sig.stop, color: down, lineStyle: 2, title: "停損" });
  (sig?.targets || []).slice(0, 2).forEach((t) => k.createPriceLine({ price: t.price, color: up, lineStyle: 2, title: t.label.slice(0, 6) }));
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-manualbuy]");
  if (b) buyDialog({ stock_id: b.dataset.manualbuy, price: +b.dataset.px, shares: 1000, stop: +b.dataset.stop, stop_reason: b.dataset.reason });
});
let sugTimer;
$("#q").addEventListener("input", (e) => {
  clearTimeout(sugTimer);
  const v = e.target.value.trim();
  if (!v) return $("#suggest").classList.add("hidden");
  sugTimer = setTimeout(async () => {
    const list = await api(`/api/search?q=${encodeURIComponent(v)}`);
    $("#suggest").innerHTML = list.map((x) => `<div data-sid="${x.stock_id}">${x.stock_id} ${esc(x.name)} <span class="muted">${esc(x.industry)}</span></div>`).join("");
    $("#suggest").classList.toggle("hidden", !list.length);
  }, 200);
});
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing && e.target.value.trim()) { $("#suggest").classList.add("hidden"); openStock(e.target.value.trim()); } });
document.addEventListener("click", (e) => { if (!e.target.closest(".toolbar")) $("#suggest").classList.add("hidden"); });

// ---------------- 勝率 ----------------
loaders.winrate = async () => {
  loadWrAll().catch(() => {});
  loadMl().catch(() => {});
  const w = await api("/api/winrate");
  $("#wr-status").textContent = w.status?.running ? `訓練中：${w.status.message}` : (w.trained ? `最後訓練：${w.trained_at.replace("T", " ")}` : "尚未訓練");
  if (!w.trained) { $("#wr-body").innerHTML = `<p class="muted">按「重新訓練」用歷史資料計算各條件組合的勝率（約 10 秒）。</p>`; return; }
  const p = w.params, best = w.best;
  const row = (r) => `<tr><td>${esc(r.names ? r.names.join(" + ") : r.name)}</td><td class="num"><b class="${r.exp > 0 ? "up" : "down"}">${r.exp > 0 ? "+" : ""}${r.exp}%</b></td>
    <td class="num">${Math.round(r.rate * 100)}%</td><td class="num">${r.train_rate != null ? Math.round(r.train_rate * 100) + "%" : ""}</td><td class="num">${r.test_rate != null ? Math.round(r.test_rate * 100) + "%" : ""}</td>
    <td class="num">${r.n.toLocaleString()}</td></tr>`;
  const head = `<tr><th>條件組合</th><th class="num">期望報酬</th><th class="num">勝率</th><th class="num">訓練勝率</th><th class="num">驗證勝率</th><th class="num">樣本</th></tr>`;
  $("#wr-body").innerHTML = `
    <div class="grid tiles">
      <div class="tile"><div class="k">全市場平均勝率</div><div class="v">${Math.round(w.base_rate.all * 100)}%</div><div class="muted">每筆期望 ${w.base_exp > 0 ? "+" : ""}${w.base_exp}%</div></div>
      <div class="tile"><div class="k">兩期都最穩的組合</div><div class="v">${best ? "+" + Math.min(best.train_exp, best.test_exp) + "%" : "—"}</div><div class="muted">${best ? esc(best.names.join(" + ")) + `（勝率 ${Math.round(best.rate * 100)}%）` : ""}</div></div>
      <div class="tile"><div class="k">達推薦門檻的組合</div><div class="v">${w.buy_count}</div><div class="muted">兩期期望 ≥ +${p.buy_exp}%、勝率 ≥ ${Math.round(p.min_rate * 100)}%</div></div>
      <div class="tile"><div class="k">期望報酬 ≤ ${p.avoid_exp}% 的組合</div><div class="v">${w.avoid_count}</div><div class="muted">共 ${w.total.toLocaleString()} 種組合</div></div>
      <div class="tile"><div class="k">樣本</div><div class="v">${(w.samples.train + w.samples.test).toLocaleString()}</div><div class="muted">${w.samples.stocks} 檔，${w.period.from}～${w.period.to}</div></div>
    </div>
    <div class="card"><p>勝率定義：訊號日隔天開盤買進，依統一出場規則（${esc(p.exit_desc || "")}）出場，扣成本後有賺算贏。
      訓練期 ${w.period.from}～${w.period.split}、驗證期 ${w.period.split}～${w.period.to}（驗證期是模型沒看過的資料）。</p>
      <p>推薦門檻：兩期都要 勝率 ≥ ${Math.round(p.min_rate * 100)}%、每筆期望 > +${p.buy_exp}%、虧損超過 6% 的比例 ≤ ${(p.max_big * 100).toFixed(0)}%（高勝率、小賺可以，但不能大賠）。</p>
      ${!w.buy_count ? `<p class="warn">目前沒有條件組合達到推薦門檻，可以到設定頁調整。</p>` : ""}</div>
    ${w.buy_rules.length ? `<div class="card"><h3>達推薦門檻的組合</h3><div class="scroll"><table>${head}${w.buy_rules.map(row).join("")}</table></div></div>` : ""}
    <div class="card"><h3>期望報酬最高的條件組合（以兩期中較差的一期排序）</h3><div class="scroll"><table>${head}${w.top.map(row).join("")}</table></div></div>
    <div class="card"><h3>單一條件的勝率</h3><div class="scroll"><table>${head}${w.single.map(row).join("")}</table></div></div>
    <div class="card"><h3>期望報酬最低的條件組合（避開）</h3><div class="scroll"><table>${head}${w.worst.map(row).join("")}</table></div></div>`;
};
// 全部個股期望報酬表
let wrAll = [], wrSort = { key: "exp", dir: -1 }, wrLimit = 100;
const DEC = { buy: ["推薦", "up"], neutral: ["中間", ""], avoid: ["不推薦", "down"] };
async function loadWrAll() {
  const d = await api("/api/winrate/all");
  if (!d.trained) { $("#wr-all").classList.add("hidden"); return; }
  $("#wr-all").classList.remove("hidden");
  wrAll = d.rows;
  const liq = wrAll.filter((r) => r.liquid);
  $("#wr-all-meta").textContent = `｜${d.date} 收盤｜${wrAll.length} 檔（流動性足夠 ${liq.length} 檔）｜推薦 ${wrAll.filter((r) => r.decision === "buy").length}・不推薦 ${wrAll.filter((r) => r.decision === "avoid").length}`;
  renderWrAll();
}
function renderWrAll() {
  const q = $("#wr-q").value.trim(), f = $("#wr-filter").value;
  let rows = wrAll.filter((r) => (f === "all" || (f === "liquid" ? r.liquid : r.decision === f))
    && (!q || `${r.stock_id}${r.name}${r.industry}`.includes(q)));
  const k = wrSort.key;
  rows.sort((a, b) => ((a[k] ?? -1e9) > (b[k] ?? -1e9) ? 1 : (a[k] ?? -1e9) < (b[k] ?? -1e9) ? -1 : 0) * wrSort.dir);
  const th = (key, label, num) => `<th class="${num ? "num " : ""}sortable" data-sort="${key}">${label}${wrSort.key === key ? (wrSort.dir < 0 ? " ▼" : " ▲") : ""}</th>`;
  $("#wr-table").innerHTML = `<tr>${th("stock_id", "代號")}${th("name", "名稱")}${th("industry", "產業")}${th("close", "收盤", 1)}${th("change_pct", "漲跌", 1)}
    ${th("exp", "每筆期望報酬", 1)}${th("rate", "勝率", 1)}${th("decision", "判斷")}<th>代表條件組合</th>${th("avg_volume", "20日均量", 1)}</tr>` +
    rows.slice(0, wrLimit).map((r) => `<tr><td><a href="#" data-sid="${r.stock_id}">${r.stock_id}</a></td><td>${esc(r.name)}</td><td>${esc(r.industry)}</td>
      <td class="num">${fmt(r.close)}</td><td class="num">${pct(r.change_pct)}</td>
      <td class="num"><b class="${r.exp > 0 ? "up" : "down"}">${r.exp > 0 ? "+" : ""}${r.exp}%</b></td><td class="num">${Math.round(r.rate * 100)}%</td>
      <td class="${DEC[r.decision][1]}">${DEC[r.decision][0]}${r.liquid ? "" : '<br><span class="muted">流動性不足</span>'}</td>
      <td class="wrap muted">${esc(r.rule)}${r.n_rules > 1 ? `（綜合 ${r.n_rules} 組）` : ""}</td><td class="num">${fmt(r.avg_volume, 0)}</td></tr>`).join("");
  $("#wr-more").classList.toggle("hidden", rows.length <= wrLimit);
  $("#wr-more").textContent = `顯示更多（還有 ${Math.max(rows.length - wrLimit, 0)} 檔）`;
}
$("#wr-table").addEventListener("click", (e) => {
  const h = e.target.closest("[data-sort]");
  if (!h) return;
  wrSort = { key: h.dataset.sort, dir: wrSort.key === h.dataset.sort ? -wrSort.dir : (["stock_id", "name", "industry"].includes(h.dataset.sort) ? 1 : -1) };
  renderWrAll();
});
$("#wr-q").addEventListener("input", () => { wrLimit = 100; renderWrAll(); });
$("#wr-filter").addEventListener("change", () => { wrLimit = 100; renderWrAll(); });
$("#wr-more").addEventListener("click", () => { wrLimit += 200; renderWrAll(); });
async function loadMl() {
  const d = await api("/api/ml");
  const m = d.model, st = d.status, ds = d.data_status, h = d.history, cv = d.coverage;
  const pg = h.progress;
  $("#ml-cov").innerHTML = `資料涵蓋（目標起點 ${cv.start_target}）：日 K ${cv.prices.from}～${cv.prices.to}（${cv.prices.days} 天）・三大法人 ${cv.insti.days} 天・融資券 ${cv.margin.days} 天・加權指數自 ${cv.index.from}`
    + ((h.running || h.other_running) && pg ? `<br>⏳ 歷史回補中：${pg.done}/${pg.total}（目前 ${pg.at}），完成後請按「重新訓練 AI 模型」與上方「重新訓練」` : "");
  $("#ml-status").textContent = [st.running ? `訓練中：${st.message}` : "", ds.running ? `資料回補：${ds.message}（${ds.done}/${ds.total}）` : "",
    d.data ? `已有法人資料 ${d.data.chips_daily || 0} 天、融資券 ${d.data.margin_daily || 0} 天` : ""].filter(Boolean).join("｜");
  if (!m) { $("#ml-body").innerHTML = `<p class="muted">尚未訓練。先按「回補法人／融資券資料」（約 30 分鐘），再按「重新訓練 AI 模型」。</p>`; return; }
  $("#ml-meta").textContent = `${m.trained_at.replace("T", " ").slice(0, 16)} 訓練｜${m.period.from}～${m.period.to}｜${m.samples.toLocaleString()} 筆、${m.stocks} 檔`;
  const th = m.threshold, best = th.best;
  const row = (t) => `<tr class="${t.pass ? "up" : ""}"><td>${t.quantile === 0 ? "全部主訊號" : `前 ${Math.round((1 - t.quantile) * 100)}%`}</td>
    <td class="num">${t.select.n.toLocaleString()}</td><td class="num">${t.select.rate == null ? "—" : Math.round(t.select.rate * 100) + "%"}</td><td class="num">${pctS(t.select.exp)}</td>
    <td class="num">${t.verify.n.toLocaleString()}</td><td class="num">${t.verify.rate == null ? "—" : Math.round(t.verify.rate * 100) + "%"}</td><td class="num">${pctS(t.verify.exp)}</td>
    <td class="num">${t.verify.big_loss_rate == null ? "—" : (t.verify.big_loss_rate * 100).toFixed(1) + "%"}</td>
    <td class="num">${t.lift.map((x) => (x == null ? "—" : (x * 100).toFixed(1) + "pp")).join(" / ")}</td><td>${t.pass ? "✅" : ""}</td></tr>`;
  $("#ml-body").innerHTML = `
    <p><b>${best ? `✅ 已啟用：每天主訊號中模型排名前 ${Math.round((1 - best.quantile) * 100)}%（且大盤多頭）建議買進` : `⚠️ 未啟用推薦：沒有任何排名門檻在兩段樣本外期間都達到「勝率 ≥ ${Math.round(m.gate.min_rate * 100)}%、期望 > +${m.gate.min_exp}%、虧損超過 6% 的比例 ≤ ${m.gate.max_big * 100}%、勝率比全體主訊號高 ${m.gate.rate_lift * 100} 個百分點」`}</b></p>
    <p class="muted">出場規則：${esc(m.params.desc || "")}</p>
    <p class="muted">全體主訊號：勝率 ${Math.round(m.base.rate * 100)}%、每筆期望 ${pctS(m.base.exp)}、平均賺 ${pctS(m.base.avg_win)}／平均賠 ${pctS(m.base.avg_loss)}、虧損超過 6% 的比例 ${(m.base.big_loss_rate * 100).toFixed(1)}%、最差 1% 平均 ${pctS(m.base.worst1)}</p>
    <div class="scroll"><table><tr><th>每日排名</th><th>選取期筆數</th><th>勝率</th><th>期望</th><th>驗證期筆數</th><th>勝率</th><th>期望</th><th>大賠比例</th><th>勝率比全體高</th><th></th></tr>${th.table.map(row).join("")}</table></div>
    <p class="muted">選取期 ${th.base[0].n.toLocaleString()} 筆／驗證期（${th.split} 起）${th.base[1].n.toLocaleString()} 筆，皆為大盤多頭日的樣本外預測。</p>
    <details><summary>Purged walk-forward 各折</summary><div class="scroll"><table><tr><th>折</th><th>訓練至</th><th>測試期</th><th>AUC</th><th>樹數</th><th>測試期全體期望</th></tr>
      ${m.folds.map((f) => `<tr><td>${f.fold}</td><td>${f.train_to}</td><td>${f.test_from}～${f.test_to}</td><td class="num">${f.auc ?? "—"}</td><td class="num">${f.best_iter}</td><td class="num">${pctS(f.base.exp)}</td></tr>`).join("")}</table></div>
      <p class="muted">AUC 0.5 = 跟亂猜一樣；每折只用測試期之前的資料訓練，並清除最後 ${m.params.horizon + 1} 天（標籤會跨進測試期）。</p></details>
    <details><summary>特徵重要度（前 25）</summary><ul class="list">${m.importance.map((x) => `<li>${esc(x.feature)}：${x.gain.toLocaleString()}</li>`).join("")}</ul></details>`;
}
$("#ml-train").addEventListener("click", async () => { await api("/api/jobs/ml_train", { method: "POST" }); toast("開始訓練 AI 模型（約 1～3 分鐘）"); setTimeout(loadMl, 2000); });
$("#ml-history").addEventListener("click", async () => { await api("/api/jobs/history", { method: "POST" }); toast("開始回補到 2021（約 4～5 小時，可隨時重開，會從斷點繼續）"); setTimeout(loadMl, 2000); });
$("#ml-backfill").addEventListener("click", async () => { await api("/api/jobs/ml_backfill", { method: "POST" }); toast("開始回補（已抓過的日期會跳過）"); setTimeout(loadMl, 2000); });
$("#retrain").addEventListener("click", async () => { await api("/api/jobs/train", { method: "POST" }); toast("開始重新訓練"); setTimeout(loaders.winrate, 1500); setTimeout(loaders.winrate, 15000); });

// ---------------- 通知 ----------------
const ICON = { pre: "🔔", buy: "🟢", stop: "🔴", reduce: "🟠", target: "🎯", info: "ℹ️", market: "📊" };
let lastAlertId = 0;
const REC_STATUS = { active: ["up", "有效推薦"], bought: ["muted", "已買進"], dismissed: ["muted", "已按不買"], expired: ["muted", "已過期"], invalid: ["down", "已失效"] };
loaders.alerts = async () => {
  const list = await api("/api/alerts");
  $("#alerts").innerHTML = list.map((a) => {
    const st = a.rec_status && REC_STATUS[a.rec_status];
    const btns = [
      a.stock_id ? `<button class="ghost" data-sid="${a.stock_id}">📈 看個股 ${a.stock_id}</button>` : "",
      a.ref_type === "recommendation" && a.ref_id ? `<button class="ghost" data-gorec="${a.ref_id}" data-recstatus="${a.rec_status || "all"}">看推薦</button>` : "",
      a.ref_type === "position" ? `<button class="ghost" data-goto="portfolio">看持股</button>` : "",
    ].join("");
    return `<div class="alert ${a.read ? "" : "unread"}"><div>${ICON[a.kind] || "•"}</div>
     <div><b>${esc(a.title)}</b>${st ? ` <span class="chip ${st[0]}">${st[1]}</span>` : ""}<pre>${esc(a.message)}</pre>${btns ? `<div class="btns">${btns}</div>` : ""}</div>
     <div class="muted">${a.ts.replace("T", " ").slice(5, 16)}${a.sent_line ? "<br>已推 LINE" : ""}</div></div>`;
  }).join("") || `<p class="muted">尚無通知。</p>`;
};
document.addEventListener("click", async (e) => {
  const g = e.target.closest("[data-gorec]");
  if (g) {
    const st = g.dataset.recstatus;
    $("#rec-filter").value = [...$("#rec-filter").options].some((o) => o.value === st) ? st : "all";
    show("recs");
    await loaders.recs();
    const card = document.querySelector(`#recs [data-recid="${g.dataset.gorec}"]`);
    if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" }); card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 2000); }
  }
  const t = e.target.closest("[data-goto]");
  if (t) show(t.dataset.goto);
});
$("#mark-read").addEventListener("click", async () => { await api("/api/alerts/read", { method: "POST" }); pollAlerts(); loaders.alerts(); });
$("#ask-notify").addEventListener("click", () => Notification.requestPermission().then((p) => toast(p === "granted" ? "已開啟瀏覽器通知" : "未允許通知")));
async function pollAlerts() {
  try {
    const list = await api("/api/alerts?limit=50");
    const unread = list.filter((a) => !a.read).length;
    $("#badge").textContent = unread; $("#badge").classList.toggle("hidden", !unread);
    const fresh = list.filter((a) => a.id > lastAlertId);
    if (lastAlertId && fresh.length && "Notification" in window && Notification.permission === "granted") {
      fresh.slice(0, 3).forEach((a) => new Notification(a.title, { body: a.message }));
    }
    if (list.length) lastAlertId = Math.max(lastAlertId, list[0].id);
    if (fresh.length && current === "alerts") loaders.alerts();
  } catch { /* 離線時略過 */ }
}
setInterval(pollAlerts, 15000);

// ---------------- 設定 ----------------
loaders.settings = async () => {
  loadAiKey();
  const s = await api("/api/settings");
  for (const [k, v] of Object.entries(s)) { const i = $(`#settings-form [name=${k}]`); if (i) { if (i.type === "checkbox") i.checked = !!v; else i.value = v; } }
  $("#exit-desc").textContent = s.exit_desc || "—";
  $("#max-stop").innerHTML = `換算：進場點到停損點的距離不能超過 ${s.max_stop_pct}%，超過就不進場。` + (s.warnings || []).map((w) => `<br><span class="warn">⚠️ ${esc(w)}</span>`).join("");
  await loadStatus();
};
async function loadStatus() {
  const st = await api("/api/status");
  const f = st.fundamentals || {};
  const fundLine = f.running ? `基本面更新中 ${f.done}/${f.total}${f.message ? `（${esc(f.message)}）` : ""}`
    : f.total ? `基本面 ${f.done}/${f.total} ${esc(f.message || "")}` : "";
  $("#sys").innerHTML = `<table>
    <tr><th>最新資料日</th><td>${st.latest_date || "尚無資料"}</td></tr>
    <tr><th>資料同步</th><td>日 K：${st.sync.running ? `進行中 ${st.sync.done}/${st.sync.total}（${esc(st.sync.message)}）` : esc(st.sync.message === "idle" ? "已是最新" : st.sync.message)}${fundLine ? `<br>${fundLine}` : ""}</td></tr>
    <tr><th>盤中監控</th><td>${st.in_session ? "🟢 盤中" : "收盤"}</td></tr></table>`;
};
// 設定：修改後自動儲存到後端資料庫（不必按儲存，也不受瀏覽器步進值限制）
let saveTimer, dirty = false;
const saveState = (t, cls = "muted") => { const el = $("#save-state"); el.textContent = t; el.className = cls; };
async function saveSettings() {
  clearTimeout(saveTimer);
  const f = $("#settings-form"), body = {};
  for (const i of f.querySelectorAll("input")) {
    if (i.type === "checkbox") body[i.name] = i.checked;
    else if (i.value.trim() !== "") body[i.name] = Number(i.value.replace(/,/g, ""));
  }
  saveState("儲存中…");
  try {
    const r = await api("/api/settings", { method: "POST", body });
    dirty = false;
    saveState(`已儲存到資料庫 ✓ ${new Date().toLocaleTimeString("zh-TW", { hour12: false })}`);
    $("#exit-desc").textContent = s.exit_desc || "—";
  $("#max-stop").innerHTML = `換算：進場點到停損點的距離不能超過 ${r.max_stop_pct}%，超過就不進場。` + (r.warnings || []).map((w) => `<br><span class="warn">⚠️ ${esc(w)}</span>`).join("");
  } catch (err) {
    let msg = err.message; try { msg = JSON.parse(msg).detail; } catch {}
    saveState(`未儲存：${Array.isArray(msg) ? msg.map((m) => m.msg).join("；") : msg}`, "danger");
  }
}
$("#settings-form").addEventListener("input", () => { dirty = true; saveState("尚未儲存…"); clearTimeout(saveTimer); saveTimer = setTimeout(saveSettings, 800); });
$("#settings-form").addEventListener("change", () => { if (dirty) saveSettings(); });
$("#settings-form").addEventListener("submit", (e) => { e.preventDefault(); saveSettings(); });
$("#reset-settings").addEventListener("click", () => dialog(`<h3>重新設定</h3>
    <p>把資金控管、選股母體、融資參數全部恢復成預設值：</p>
    <ul class="list"><li>帳戶資金 1,000,000、每筆部位 100,000、單筆最大虧損 10,000（講義範例）</li>
    <li>手續費 3 折、20 日均量 1,000 張、最低股價 10 元</li>
    <li>上市融資 6 成、上櫃 5 成、年利率 6.5%、追繳維持率 130%</li></ul>
    <p class="muted">只會重設參數，不會刪除持股、交易紀錄、推薦與通知。重設後可以按「復原上次的設定」改回來。</p>`,
  async () => {
    clearTimeout(saveTimer); dirty = false;
    const r = await api("/api/settings/reset", { method: "POST" });
    await loaders.settings();
    $("#undo-reset").classList.toggle("hidden", !r.can_undo);
    saveState("已恢復預設值 ✓");
  }));
$("#undo-reset").addEventListener("click", async () => {
  await api("/api/settings/undo-reset", { method: "POST" });
  $("#undo-reset").classList.add("hidden");
  await loaders.settings(); saveState("已復原上次的設定 ✓");
});
window.addEventListener("beforeunload", (e) => { if (dirty) { saveSettings(); e.preventDefault(); e.returnValue = ""; } });
$$("[data-job]").forEach((b) => b.addEventListener("click", async () => {
  const r = await api(`/api/jobs/${b.dataset.job}`, { method: "POST" });
  toast(b.dataset.job === "test_line" ? (r.ok ? "LINE 推播成功" : r.configured ? "LINE 推播失敗，請檢查 token" : "尚未設定 LINE") : "已開始執行");
  setTimeout(loaders.settings, 1500);
}));

// ---------------- 啟動 ----------------
setInterval(() => { if (current === "market") loaders.market(); if (current === "portfolio") loaders.portfolio(); }, 30000);
setInterval(() => { if (current === "settings") loadStatus().catch(() => {}); }, 10000);
pollAlerts();
window.addEventListener("hashchange", () => { const t = location.hash.slice(1); if (t && t !== current && loaders[t]) show(t); });
show(location.hash.slice(1) || "market");

// ---------------- AI 助理 ----------------
const chat = { id: null, busy: false };
try { chat.id = localStorage.getItem("chat_id"); } catch {}
function mdInline(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
}
function md(src) {
  // 精簡 Markdown：段落、標題、條列、表格、粗體、行內程式碼（先跳脫 HTML 再轉換）
  const lines = src.split("\n"), out = [];
  let list = null, table = null;
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } if (table) { out.push("</table></div>"); table = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\|.*\|$/.test(line.trim())) {
      const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue;
      if (!table) { close(); out.push('<div class="scroll"><table>'); table = "head"; out.push(`<tr>${cells.map((c) => `<th>${mdInline(c)}</th>`).join("")}</tr>`); table = "body"; continue; }
      out.push(`<tr>${cells.map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`);
      continue;
    }
    let m;
    if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${mdInline(m[1])}</li>`); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${mdInline(m[1])}</li>`); continue; }
    close();
    if ((m = line.match(/^#{1,6}\s+(.*)/))) out.push(`<h4>${mdInline(m[1])}</h4>`);
    else if (line.trim() === "---") continue;
    else if (line.trim()) out.push(`<p>${mdInline(line)}</p>`);
  }
  close();
  return out.join("");
}
function chatContext() {
  const names = { market: "大盤頁", recs: "推薦頁", portfolio: "持股健檢頁", stock: "個股分析頁", winrate: "勝率頁", alerts: "通知頁", settings: "設定頁" };
  let c = names[current] || current;
  const sid = $("#q")?.value.trim();
  if (current === "stock" && sid) c += `，正在看 ${sid}`;
  return c;
}
function chatAdd(cls, html) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.innerHTML = html;
  $("#chat-log").appendChild(el);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return el;
}
async function chatSend(text) {
  text = text.trim();
  if (!text || chat.busy) return;
  chat.busy = true; $("#chat-send").disabled = true;
  $(".chat-hello")?.remove();
  chatAdd("user", esc(text));
  const box = chatAdd("ai", `<div class="tools"></div><div class="body typing"></div>`);
  const body = box.querySelector(".body"), tools = box.querySelector(".tools");
  let acc = "";
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, conversation_id: chat.id, context: chatContext() }) });
    if (!r.ok) throw new Error(await r.text());
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!chunk.startsWith("data: ")) continue;
        const ev = JSON.parse(chunk.slice(6));
        if (ev.type === "conv") { chat.id = ev.id; try { localStorage.setItem("chat_id", ev.id); } catch {} }
        else if (ev.type === "text") { acc += ev.text; body.innerHTML = md(acc); }
        else if (ev.type === "tool") { tools.insertAdjacentHTML("beforeend", `<span>🔎 ${esc(ev.label)}</span>`); if (acc) { acc += "\n\n"; } }
        else if (ev.type === "error") chatAdd("err", esc(ev.message));
        $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
      }
    }
  } catch (e) {
    chatAdd("err", esc(`連線失敗：${e.message}`));
  } finally {
    body.classList.remove("typing");
    if (!acc && !tools.children.length) box.remove();
    chat.busy = false; $("#chat-send").disabled = false; $("#chat-input").focus();
  }
}
function chatOpen(open) {
  $("#chat").classList.toggle("hidden", !open);
  $("#chat-fab").classList.toggle("hidden", open);
  if (open) {
    $("#chat-input").focus();
    api("/api/chat/status").then((s) => {
      $("#chat-model").textContent = s.configured ? s.model : "尚未設定 API 金鑰";
    }).catch(() => {});
  }
}
$("#chat-fab").addEventListener("click", () => chatOpen(true));
$("#chat-close").addEventListener("click", () => chatOpen(false));
$("#chat-new").addEventListener("click", async () => {
  if (chat.busy) return;
  if (chat.id) api(`/api/chat/${chat.id}/reset`, { method: "POST" }).catch(() => {});
  chat.id = null; try { localStorage.removeItem("chat_id"); } catch {}
  $("#chat-log").innerHTML = `<div class="msg ai"><p>已開始新的對話。</p></div>`;
});
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#chat-input").value; $("#chat-input").value = ""; chatSend(t); });
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#chat-form").requestSubmit(); }
});
$("#chat-input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`; });
$("#chat-log").addEventListener("click", (e) => { const b = e.target.closest("[data-ask]"); if (b) chatSend(b.dataset.ask); });

// ---------------- AI 金鑰設定 ----------------
function renderAiKey(s) {
  $("#ai-key-state").innerHTML = s.configured
    ? `✅ 已設定（${esc(s.key_hint || "")}），模型 ${esc(s.model)}`
    : `尚未設定，AI 助理暫時無法使用`;
  $("#ai-key-del").classList.toggle("hidden", !s.configured);
  if (!$("#chat").classList.contains("hidden")) $("#chat-model").textContent = s.configured ? s.model : "尚未設定 API 金鑰";
}
function loadAiKey() { api("/api/chat/status").then(renderAiKey).catch(() => { $("#ai-key-state").textContent = "無法讀取狀態"; }); }
$("#ai-key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const key = $("#ai-key").value.trim();
  if (!key) return;
  const msg = $("#ai-key-msg");
  msg.textContent = "驗證中…"; msg.className = "muted";
  try {
    const r = await fetch("/api/settings/ai-key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "儲存失敗");
    $("#ai-key").value = "";
    msg.textContent = "已驗證並儲存，可以開始問 AI 了"; msg.className = "up";
    renderAiKey(d);
  } catch (err) { msg.textContent = err.message; msg.className = "danger"; }
});
$("#ai-key-del").addEventListener("click", async () => {
  if (!confirm("確定要移除 API 金鑰嗎？移除後 AI 助理將無法使用。")) return;
  renderAiKey(await api("/api/settings/ai-key", { method: "DELETE" }));
  $("#ai-key-msg").textContent = "已移除"; $("#ai-key-msg").className = "muted";
});
