"""基本面健檢：把《從0到1選出一家好公司》的財報判讀寫成檢核項目。

四大重點：獲利能力、成長性、經營效率、償債能力，再加上「控制錢」的現金流型態。
比較方式：跟自己以往同期比。
"""
from collections import defaultdict

import pandas as pd


def _pivot(rows: list[dict]) -> pd.DataFrame:
    """FinMind 長表 (date, type, value) → 寬表 index=date。"""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[~df["type"].str.endswith("_per")]
    wide = df.pivot_table(index="date", columns="type", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _decumulate(cf: pd.DataFrame) -> pd.DataFrame:
    """台灣現金流量表為年初累計數，轉回單季。"""
    if cf.empty:
        return cf
    out = cf.copy()
    for _, grp in cf.groupby(cf.index.year):
        single = grp.diff()
        first = grp.index[0]
        # 年度第一筆若是 Q1 即為單季；若缺 Q1 則無法還原，設為 NaN
        single.loc[first] = grp.loc[first] if first.quarter == 1 else float("nan")
        out.loc[grp.index] = single
    return out


def _g(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(dtype=float, index=df.index)


def _pct(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return (a - b) / abs(b) * 100


def analyze(income: list[dict], balance: list[dict], cashflow: list[dict], revenue: list[dict],
            per: list[dict] | None = None) -> dict:
    inc, bal, cf = _pivot(income), _pivot(balance), _decumulate(_pivot(cashflow))
    checks: list[dict] = []
    score = 40.0
    fatal = False

    def add(cat, name, status, detail, pts=0.0, is_fatal=False):
        nonlocal score, fatal
        checks.append({"category": cat, "name": name, "status": status, "detail": detail})
        score += pts
        fatal = fatal or is_fatal

    metrics: dict = {}

    # ---------- 獲利能力 ----------
    if not inc.empty and len(inc) >= 4:
        rev_q, gp_q = _g(inc, "Revenue"), _g(inc, "GrossProfit")
        ni_q = _g(inc, "IncomeAfterTaxes")
        metrics["revenue_ttm"] = float(rev_q.iloc[-4:].sum())
        gm =(gp_q / rev_q * 100).dropna()
        if len(gm) >= 4:
            recent, older = gm.iloc[-4:].mean(), gm.iloc[-8:-4].mean() if len(gm) >= 8 else gm.iloc[:-4].mean()
            metrics["gross_margin"] = round(float(gm.iloc[-1]), 2)
            if pd.isna(older) or recent >= older - 1:
                add("獲利能力", "毛利率長期穩定或上升", "good",
                    f"近四季平均 {recent:.1f}%（前四季 {older:.1f}%）— 護城河、議價能力的訊號", 8)
            else:
                add("獲利能力", "毛利率下滑", "warn", f"近四季平均 {recent:.1f}% < 前四季 {older:.1f}%，可能被削價競爭", -6)
        nm = (ni_q / rev_q * 100).dropna()
        if len(nm):
            metrics["net_margin"] = round(float(nm.iloc[-1]), 2)
            if len(gm) and gm.iloc[-1] < 10 and rev_q.iloc[-4:].sum() > 0:
                add("獲利能力", "高營收低毛利", "warn", f"毛利率僅 {gm.iloc[-1]:.1f}%，大利多也可能反應較弱", -4)
        eps = _g(inc, "EPS").dropna()
        if len(eps) >= 8:
            ttm, prev_ttm = eps.iloc[-4:].sum(), eps.iloc[-8:-4].sum()
            metrics["eps_ttm"] = round(float(ttm), 2)
            ch = _pct(ttm, prev_ttm)
            add("獲利能力", "EPS 跟自己比", "good" if ttm > prev_ttm else "warn",
                f"近四季 EPS {ttm:.2f}（前四季 {prev_ttm:.2f}，{'+' if (ch or 0) >= 0 else ''}{ch or 0:.0f}%）",
                6 if ttm > prev_ttm else -4)
        if len(ni_q) >= 5 and ni_q.iloc[-1] > 0 and ni_q.iloc[-5] < 0:
            metrics["turnaround"] = True
            add("獲利能力", "由虧轉盈", "good", "最新一季轉為獲利、去年同季虧損——可能的黑馬（本夢比）", 4)
        if ni_q.iloc[-4:].sum() <= 0:
            add("獲利能力", "近四季沒有在賺錢", "bad", "淘汰沒在賺錢、體質爛的公司", -20, is_fatal=True)

        # ROE / ROA（槓桿陷阱）
        if not bal.empty:
            eq = _g(bal, "EquityAttributableToOwnersOfParent").dropna()
            ta = _g(bal, "TotalAssets").dropna()
            ni_ttm = ni_q.iloc[-4:].sum()
            if len(eq) and len(ta):
                roe = ni_ttm / eq.iloc[-4:].mean() * 100
                roa = ni_ttm / ta.iloc[-4:].mean() * 100
                metrics.update(roe=round(float(roe), 2), roa=round(float(roa), 2))
                if roe >= 15 and roa < 5:
                    add("獲利能力", "ROE 高、ROA 低", "warn",
                        f"ROE {roe:.1f}% / ROA {roa:.1f}%：獲利可能大多來自高財務槓桿", -6)
                elif roe >= 10:
                    add("獲利能力", "ROE 為股東賺回的獲利", "good", f"ROE {roe:.1f}% / ROA {roa:.1f}%", 6)
                else:
                    add("獲利能力", "ROE 偏低", "info", f"ROE {roe:.1f}% / ROA {roa:.1f}%", -2)
    else:
        add("獲利能力", "財報資料不足", "info", "季報少於四季，無法判斷", 0)

    # ---------- 成長性 ----------
    if revenue:
        rv = pd.DataFrame(revenue)
        rv["ym"] = rv["revenue_year"] * 100 + rv["revenue_month"]
        rv = rv.sort_values("ym").drop_duplicates("ym").set_index("ym")["revenue"]
        yoys, moms = [], []
        for ym in rv.index[-3:]:
            last_y = ym - 100
            if last_y in rv.index:
                yoys.append(_pct(rv[ym], rv[last_y]))
        if len(rv) >= 2:
            moms.append(_pct(rv.iloc[-1], rv.iloc[-2]))
        if yoys:
            avg = sum(y for y in yoys if y is not None) / len(yoys)
            metrics["revenue_yoy_3m"] = round(avg, 1)
            metrics["revenue_mom"] = None if not moms or moms[0] is None else round(moms[0], 1)
            add("成長性", "月營收年增率（近三月平均）", "good" if avg > 10 else ("info" if avg > 0 else "warn"),
                f"YoY {avg:.1f}%，MoM {metrics['revenue_mom']}%", 8 if avg > 10 else (2 if avg > 0 else -6))
    if not inc.empty and len(inc) >= 3:
        rq = _g(inc, "Revenue").dropna()
        qoq = [(_pct(rq.iloc[i], rq.iloc[i - 1])) for i in range(len(rq) - 3, len(rq)) if i > 0]
        ups = sum(1 for q in qoq if q is not None and q > 0)
        metrics["revenue_qoq"] = [None if q is None else round(float(q), 1) for q in qoq]
        if ups >= 2:
            add("成長性", "季營收穩定成長（QoQ）", "good", f"近三季 QoQ：{metrics['revenue_qoq']}——會受到資金信賴", 6)

    # ---------- 現金流（公司要會控制錢） ----------
    if not cf.empty and len(cf) >= 4:
        ocf = _g(cf, "CashFlowsFromOperatingActivities").dropna()
        icf = _g(cf, "CashProvidedByInvestingActivities").dropna()
        fcf = _g(cf, "CashFlowsProvidedFromFinancingActivities").dropna()
        if len(ocf) >= 4 and len(icf) >= 4 and len(fcf) >= 4:
            o, i, f = ocf.iloc[-4:].sum(), icf.iloc[-4:].sum(), fcf.iloc[-4:].sum()
            o_prev = ocf.iloc[-8:-4].sum() if len(ocf) >= 8 else None
            metrics["cashflow_ttm"] = {"operating": o, "investing": i, "financing": f}
            growing = o_prev is None or o >= o_prev
            if o > 0 and i < 0 and f < 0:
                add("現金流", "財務穩定的好公司型態", "good",
                    "營業現金流入、投資為負（持續擴廠）、籌資為負（還債／發股利）", 12)
            elif (not growing or o <= 0) and f > 0 and i > 0:
                add("現金流", "危險型態", "bad", "營業沒成長、籌資為正、投資為正：沒賺錢、借錢並變賣資產", -25, is_fatal=True)
            elif o <= 0 and i < 0 and f > 0:
                add("現金流", "擴張陣痛期", "warn", "營業、投資為負、籌資為正：把資金拿去擴產，營業轉正前風險極大，不建議大部位", -8)
            elif o <= 0:
                add("現金流", "營業現金流為負", "warn", "資金可能在某一階段卡住，觀察是否順利消化", -8)
            else:
                add("現金流", "現金流尚可", "info", f"營業 {o/1e8:.1f} 億／投資 {i/1e8:.1f} 億／籌資 {f/1e8:.1f} 億", 2)
            if i > 0 and o > 0:
                add("現金流", "投資活動現金流為正", "info", "變賣土地廠房要問用途：轉型還是缺錢？", 0)

    # ---------- 償債能力 ----------
    if not bal.empty:
        last = bal.iloc[-1]
        ca, cl = last.get("CurrentAssets"), last.get("CurrentLiabilities")
        contract = sum(last.get(k, 0) or 0 for k in ("CurrentContractLiabilities", "ContractLiabilitiesCurrent"))
        inv = last.get("Inventories") or 0
        prepaid = last.get("Prepayments") or 0
        if ca and cl:
            cl_adj = max(cl - contract, 1)  # 合約負債不需要清償，排除
            cr = ca / cl_adj * 100
            qr = (ca - inv - prepaid) / cl_adj * 100
            metrics.update(current_ratio=round(cr, 1), quick_ratio=round(qr, 1))
            if cr >= 200:
                add("償債能力", "流動比率 ≥ 200%", "good", f"{cr:.0f}%（已排除合約負債）", 4)
            elif cr < 100:
                add("償債能力", "流動比率 < 100%", "warn", f"{cr:.0f}%：短期還債壓力大", -6)
            if qr < 80:
                add("償債能力", "速動比率偏低", "warn", f"{qr:.0f}%：常買到被套牢一輩子的股票，從速動比率就能避開很大的風險", -6)
            else:
                add("償債能力", "速動比率", "good" if qr >= 100 else "info", f"{qr:.0f}%", 3 if qr >= 100 else 0)
        tl, te = last.get("Liabilities"), last.get("Equity")
        if tl and te:
            de = tl / te
            metrics["debt_equity"] = round(de, 2)
            add("償債能力", "負債比率（負債/權益）", "info" if de <= 1 else "warn",
                f"{de:.2f}：" + ("風險較小" if de < 1 else "槓桿高，只要跟得上成長就沒問題"), 0 if de <= 1.5 else -3)

        # ---------- 經營效率／盈餘品質 ----------
        if len(bal) >= 5 and not inc.empty and len(inc) >= 5:
            ar = _g(bal, "AccountsReceivableNet").dropna()
            inv_s = _g(bal, "Inventories").dropna()
            rq = _g(inc, "Revenue").dropna()
            rev_yoy = _pct(rq.iloc[-1], rq.iloc[-5]) if len(rq) >= 5 else None
            if len(ar) >= 5 and rev_yoy is not None:
                ar_yoy = _pct(ar.iloc[-1], ar.iloc[-5])
                if ar_yoy is not None and ar_yoy - rev_yoy > 30:
                    add("經營效率", "應收帳款異常增加", "warn",
                        f"應收 YoY {ar_yoy:.0f}% 遠高於營收 YoY {rev_yoy:.0f}%：小心虛增資產、呆帳", -6)
            if len(inv_s) >= 5 and rev_yoy is not None:
                inv_yoy = _pct(inv_s.iloc[-1], inv_s.iloc[-5])
                if inv_yoy is not None and inv_yoy - rev_yoy > 40:
                    add("經營效率", "存貨異常增加", "warn",
                        f"存貨 YoY {inv_yoy:.0f}% vs 營收 YoY {rev_yoy:.0f}%：可能賣不掉，也可能是 AI 備貨（上游吃緊）", -3)

    # ---------- 本益比 ----------
    if per:
        p = per[-1]
        metrics["per"] = p.get("PER")
        metrics["pbr"] = p.get("PBR")
        metrics["dividend_yield"] = p.get("dividend_yield")

    score = max(0.0, min(100.0, score))
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    return {
        "score": round(score, 1),
        "grade": grade,
        "pass": (not fatal) and score >= 55,
        "fatal": fatal,
        "checks": checks,
        "metrics": metrics,
    }


def classify_peers(reports: dict[str, dict], industry_of: dict[str, str]) -> dict[str, list[str]]:
    """講義總結：體質合格後分三類——穩定成長的龍頭、最賺錢的公司、最有本夢比的黑馬。"""
    by_ind: dict[str, list[str]] = defaultdict(list)
    for sid, rep in reports.items():
        if rep.get("pass"):
            by_ind[industry_of.get(sid, "其他")].append(sid)
    tags: dict[str, list[str]] = defaultdict(list)
    for _, sids in by_ind.items():
        if not sids:
            continue
        rev = {s: reports[s]["metrics"].get("revenue_ttm", 0) or 0 for s in sids}
        margin = {s: reports[s]["metrics"].get("net_margin", -999) or -999 for s in sids}
        tags[max(rev, key=rev.get)].append("族群龍頭")
        tags[max(margin, key=margin.get)].append("最賺錢")
        for s in sids:
            if reports[s]["metrics"].get("turnaround"):
                tags[s].append("黑馬")
    return tags
