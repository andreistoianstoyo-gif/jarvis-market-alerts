# Unelte personalizate JARVIS — Screener Global, Backtest & Evaluare Companii
import os
import re
import json
import math
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))
FISIER_WATCHLIST_REPO = os.path.join(DIRECTOR_CURENT, "watchlist.json")
FISIER_WATCHLIST_LOCAL = os.path.expanduser("~/.jarvis/watchlist_trading.json")

WATCHLIST_IMPLICIT = [
    # Tech, AI & Semiconductori (inclusiv mid/small-cap)
    "NVDA", "AMD", "PLTR", "SMCI", "ARM", "MRVL", "IONQ", "RGTI", "SOUN", "PATH",
    # Energie, Uraniu & Infrastructură AI
    "CCJ", "UEC", "OKLO", "SMR", "VST", "CEG", "ENPH", "FSLR",
    # Biotech, MedTech & Nișă
    "CRSP", "NTLA", "RXRX", "HIMS", "TEM", "ISRG",
    # Apărare, Spațiu & Robotică
    "RKLB", "LUNR", "ASTS", "KTOS", "AVAV", "SYM",
    # BVB România
    "TLV.RO", "SNP.RO", "H2O.RO", "SNG.RO", "BRD.RO", "M.RO",
]


def _incarca_watchlist() -> list[str]:
    for cale in (FISIER_WATCHLIST_REPO, FISIER_WATCHLIST_LOCAL):
        if os.path.exists(cale):
            try:
                with open(cale, "r", encoding="utf-8") as f:
                    date = json.load(f)
                    if isinstance(date, list) and date:
                        return [str(s).strip().upper() for s in date if str(s).strip()]
            except Exception:
                pass
    os.makedirs(os.path.dirname(FISIER_WATCHLIST_LOCAL), exist_ok=True)
    with open(FISIER_WATCHLIST_LOCAL, "w", encoding="utf-8") as f:
        json.dump(WATCHLIST_IMPLICIT, f, indent=2)
    return list(WATCHLIST_IMPLICIT)


def cauta_simbol_bursa(nume_companie: str) -> str:
    """Caută simbolul bursier (ticker) pentru orice companie din lume, inclusiv firme mici/de nișă."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(nume_companie)}&quotesCount=6"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).json()
        quotes = r.get("quotes", [])
        if not quotes:
            return f"Nu am găsit niciun simbol bursier pentru '{nume_companie}'."
        linii = [f"Rezultate bursiere pentru '{nume_companie}':"]
        for q in quotes:
            sim = q.get("symbol", "")
            nume = q.get("shortname") or q.get("longname") or ""
            bursa = q.get("exchDisp", "")
            tip = q.get("quoteType", "")
            if sim:
                linii.append(f"• {sim} — {nume} ({bursa}, {tip})")
        return "\n".join(linii)
    except Exception as e:
        return f"Eroare la căutarea simbolului: {e}"


def adauga_in_watchlist_trading(simboluri: str, sterge: bool = False) -> str:
    """Adaugă sau șterge simboluri bursiere (separate prin virgulă) din watchlist.json."""
    lista = _incarca_watchlist()
    noi = [s.strip().upper() for s in simboluri.split(",") if s.strip()]
    if sterge:
        lista = [s for s in lista if s not in noi]
        actiune = "Am eliminat"
    else:
        for s in noi:
            if s not in lista:
                lista.append(s)
        actiune = "Am adăugat"
    with open(FISIER_WATCHLIST_REPO, "w", encoding="utf-8") as f:
        json.dump(lista, f, indent=2)
    return f"{actiune} {', '.join(noi)}. Watchlist-ul are acum {len(lista)} acțiuni."


def _calculeaza_metrici_actiune(simbol: str) -> dict | None:
    """Descarcă istoricul pe 2 ani, calculează indicatorii tehnici, SL/TP prin ATR și rata de succes istorică."""
    try:
        df = yf.download(simbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].astype(float)

        if float(vol.tail(20).mean()) < 1000:
            return None

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        vol_ratio = float(vol.iloc[-1] / max(1.0, vol.tail(20).mean()))
        pret_curent = float(close.iloc[-1])
        pret_ieri = float(close.iloc[-2])
        variatie_zi = round(((pret_curent - pret_ieri) / max(0.01, pret_ieri)) * 100, 2)

        atr_curent = float(atr.iloc[-1])
        rsi_curent = float(rsi.iloc[-1])
        s20 = float(sma20.iloc[-1])
        s50 = float(sma50.iloc[-1])
        s200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else s50

        if np.isnan(atr_curent) or atr_curent <= 0:
            return None

        stop_loss = round(pret_curent - 1.5 * atr_curent, 2)
        tp1 = round(pret_curent + 3.0 * atr_curent, 2)
        tp2 = round(pret_curent + 4.5 * atr_curent, 2)
        risc_proc = round(((pret_curent - stop_loss) / pret_curent) * 100, 2)
        profit1_proc = round(((tp1 - pret_curent) / pret_curent) * 100, 2)

        castiguri = 0
        pierderi = 0
        i = 55
        while i < len(df) - 15:
            cond_trend = close.iloc[i] > sma50.iloc[i] and (38 <= rsi.iloc[i] <= 66)
            cond_oversold = rsi.iloc[i] < 33 and close.iloc[i] > close.iloc[i - 1]
            if cond_trend or cond_oversold:
                entry_b = float(close.iloc[i])
                atr_b = float(atr.iloc[i])
                if not np.isnan(atr_b) and atr_b > 0:
                    sl_b = entry_b - 1.5 * atr_b
                    tp_b = entry_b + 3.0 * atr_b
                    for j in range(i + 1, min(i + 25, len(df))):
                        if float(low.iloc[j]) <= sl_b:
                            pierderi += 1
                            i = j
                            break
                        if float(high.iloc[j]) >= tp_b:
                            castiguri += 1
                            i = j
                            break
            i += 1

        total_tranzactii = castiguri + pierderi
        win_rate = round((castiguri / total_tranzactii) * 100, 1) if total_tranzactii >= 3 else 50.0

        scor = 0
        motive = []
        if pret_curent > s50 > s200:
            scor += 30
            motive.append("Trend ascendent confirmat (Preț > SMA50 > SMA200)")
        elif pret_curent > s20:
            scor += 15
            motive.append("Peste media pe 20 zile")

        if 42 <= rsi_curent <= 62:
            scor += 25
            motive.append(f"Momentum sănătos (RSI={rsi_curent:.1f})")
        elif rsi_curent < 34:
            scor += 30
            motive.append(f"Supravândut cu potențial de revenire (RSI={rsi_curent:.1f})")
        elif rsi_curent > 74:
            scor -= 20
            motive.append(f"Supracumpărat (RSI={rsi_curent:.1f} — risc de corecție)")

        if vol_ratio >= 1.4:
            scor += 25
            motive.append(f"Volum neobișnuit de mare ({vol_ratio:.1f}x media)")

        if win_rate >= 55 and total_tranzactii >= 4:
            scor += 20
            motive.append(f"Istoric 2 ani favorabil ({win_rate}% win-rate din {total_tranzactii} semnale)")

        return {
            "simbol": simbol,
            "pret": round(pret_curent, 2),
            "variatie_zi": variatie_zi,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "risc_proc": risc_proc,
            "profit1_proc": profit1_proc,
            "rsi": round(rsi_curent, 1),
            "vol_ratio": round(vol_ratio, 2),
            "win_rate": win_rate,
            "nr_semnale_trecut": total_tranzactii,
            "scor": scor,
            "motive": motive,
        }
    except Exception:
        return None


def analizeaza_actiune_complet(simbol: str, capital_referinta: float = 1000.0) -> str:
    """Analizează o acțiune (inclusiv de nișă), calculează Stop-Loss, Take-Profit, mărimea poziției și rata istorică pe 2 ani."""
    sim = simbol.strip().upper()
    rez = _calculeaza_metrici_actiune(sim)
    if not rez:
        return f"Nu am putut prelua suficiente date istorice pentru '{sim}'. Încearcă mai întâi `cauta_simbol_bursa`."

    risc_maxim_bani = capital_referinta * 0.02
    risc_per_actiune = max(0.01, rez["pret"] - rez["stop_loss"])
    nr_actiuni = int(risc_maxim_bani // risc_per_actiune)
    valoare_pozitie = round(nr_actiuni * rez["pret"], 2)

    motive_str = "\n  • ".join(rez["motive"]) if rez["motive"] else "Piață neutră momentan"
    return (
        f"📊 ANALIZĂ TEHNICĂ & BACKTEST (2 ANI): {rez['simbol']} (Scor: {rez['scor']}/100)\n"
        f"• Preț actual: {rez['pret']} ({rez['variatie_zi']:+.2f}% azi) | RSI(14): {rez['rsi']} | Volum: {rez['vol_ratio']}x\n"
        f"• Istoric 2 ani pe strategie: {rez['win_rate']}% reușită ({rez['nr_semnale_trecut']} semnale testate)\n"
        f"• Niveluri calculate (ATR volatilitate):\n"
        f"  🛑 Stop-Loss: {rez['stop_loss']} (-{rez['risc_proc']}%)\n"
        f"  🎯 Take-Profit 1 (1:2): {rez['tp1']} (+{rez['profit1_proc']}%)\n"
        f"  🚀 Take-Profit 2 (1:3): {rez['tp2']}\n"
        f"• Exemplu poziționare (la capital {capital_referinta}, risc 2% = {risc_maxim_bani:.1f}): "
        f"{nr_actiuni} acțiuni (valoare ~{valoare_pozitie})\n"
        f"• Semnale detectate:\n  • {motive_str}"
    )


def scaneaza_piata_pentru_oportunitati(simboluri_extra: str = "") -> str:
    """Scanează watchlist-ul global și returnează top 5 acțiuni cu cel mai bun scor și niveluri SL/TP."""
    lista = _incarca_watchlist()
    if simboluri_extra.strip():
        for s in simboluri_extra.split(","):
            sc = s.strip().upper()
            if sc and sc not in lista:
                lista.append(sc)

    rezultate = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_calculeaza_metrici_actiune, lista):
            if r is not None:
                rezultate.append(r)

    if not rezultate:
        return "Nu am putut prelua cotațiile în acest moment (verifică conexiunea la internet)."

    rezultate.sort(key=lambda x: x["scor"], reverse=True)
    top = rezultate[:5]

    blocuri = [f"🔎 TOP 5 OPORTUNITĂȚI DIN {len(rezultate)} ACȚIUNI ANALIZATE:"]
    for idx, r in enumerate(top, 1):
        motiv_principal = r["motive"][0] if r["motive"] else "Setup tehnic echilibrat"
        blocuri.append(
            f"{idx}. **{r['simbol']}** — Scor: {r['scor']}/100 | Preț: {r['pret']} ({r['variatie_zi']:+.2f}%) | Win-rate 2Y: {r['win_rate']}%\n"
            f"   🛑 SL: {r['stop_loss']} (-{r['risc_proc']}%) | 🎯 TP1: {r['tp1']} (+{r['profit1_proc']}%) | 🚀 TP2: {r['tp2']}\n"
            f"   📌 {motiv_principal}"
        )
    return "\n\n".join(blocuri)


def _extrage_stiri_yahoo(ticker_obj) -> list[str]:
    stiri_formatate = []
    try:
        raw_news = getattr(ticker_obj, "news", None) or []
        for item in raw_news[:6]:
            if not isinstance(item, dict):
                continue
            continut = item.get("content") if isinstance(item.get("content"), dict) else item
            titlu = continut.get("title") or item.get("title") or ""
            sursa = (
                continut["provider"].get("displayName", "")
                if isinstance(continut.get("provider"), dict)
                else item.get("publisher", "")
            )
            if titlu:
                stiri_formatate.append(f"• [{sursa or 'Știre'}] {titlu.strip()}")
    except Exception:
        pass
    return stiri_formatate


def evalueaza_companie_complet(simbol: str, capital_referinta: float = 1000.0) -> str:
    """Evaluează complet o companie (date financiare fundamentale, știri recente, analiză tehnică, Stop-Loss și Take-Profit)."""
    sim = simbol.strip().upper()
    try:
        tk = yf.Ticker(sim)
        info = tk.info or {}
    except Exception as e:
        return f"Nu am putut accesa datele companiei '{sim}': {e}"

    nume_companie = info.get("longName") or info.get("shortName") or sim
    sector = info.get("sector") or "Nespecificat"
    industrie = info.get("industry") or "Nespecificată"
    moneda = info.get("currency") or "USD"

    mcap = info.get("marketCap")
    mcap_str = (
        f"{mcap / 1e9:.2f} mld. {moneda}" if isinstance(mcap, (int, float)) and mcap >= 1e9
        else (f"{mcap / 1e6:.1f} mil. {moneda}" if isinstance(mcap, (int, float)) else "N/A")
    )

    rev_growth = info.get("revenueGrowth")
    profit_margin = info.get("profitMargins")
    forward_pe = info.get("forwardPE")
    trailing_pe = info.get("trailingPE")
    debt_to_equity = info.get("debtToEquity")
    target_price = info.get("targetMeanPrice")
    recomandare_analisti = info.get("recommendationKey", "N/A").upper()

    scor_fund = 50
    argumente_fund = []

    if isinstance(rev_growth, (int, float)):
        if rev_growth > 0.20:
            scor_fund += 20
            argumente_fund.append(f"Creștere explozivă a veniturilor (+{rev_growth * 100:.1f}% YoY)")
        elif rev_growth > 0.05:
            scor_fund += 10
            argumente_fund.append(f"Venituri în creștere (+{rev_growth * 100:.1f}% YoY)")
        elif rev_growth < 0:
            scor_fund -= 20
            argumente_fund.append(f"Venituri în scădere ({rev_growth * 100:.1f}% YoY)")

    if isinstance(profit_margin, (int, float)):
        if profit_margin > 0.20:
            scor_fund += 15
            argumente_fund.append(f"Marjă de profit foarte mare ({profit_margin * 100:.1f}%)")
        elif profit_margin > 0.08:
            scor_fund += 8
            argumente_fund.append(f"Companie profitabilă (marjă {profit_margin * 100:.1f}%)")
        elif profit_margin < 0:
            scor_fund -= 15
            argumente_fund.append(f"Companie pe pierdere (marjă {profit_margin * 100:.1f}%)")

    if isinstance(debt_to_equity, (int, float)):
        if debt_to_equity < 60:
            scor_fund += 10
            argumente_fund.append(f"Datorii mici (Debt/Equity: {debt_to_equity:.1f}%)")
        elif debt_to_equity > 200:
            scor_fund -= 15
            argumente_fund.append(f"Grad ridicat de îndatorare (Debt/Equity: {debt_to_equity:.1f}%)")

    if isinstance(forward_pe, (int, float)) and 0 < forward_pe < 35:
        scor_fund += 5
        argumente_fund.append(f"Evaluare rezonabilă (Forward P/E: {forward_pe:.1f})")

    scor_fund = max(0, min(100, scor_fund))

    tehnic = _calculeaza_metrici_actiune(sim)
    bloc_tehnic = "Date tehnice insuficiente pentru calculul ATR."
    pret_ref = info.get("currentPrice") or info.get("regularMarketPrice")

    if tehnic:
        pret_ref = tehnic["pret"]
        risc_bani = capital_referinta * 0.02
        nr_actiuni = int(risc_bani // max(0.01, tehnic["pret"] - tehnic["stop_loss"]))
        bloc_tehnic = (
            f"• Scor Tehnic: {tehnic['scor']}/100 | Preț: {tehnic['pret']} {moneda} ({tehnic['variatie_zi']:+.2f}% azi) | RSI(14): {tehnic['rsi']} | Volum: {tehnic['vol_ratio']}x\n"
            f"• Win-Rate istoric (2 ani): {tehnic['win_rate']}% ({tehnic['nr_semnale_trecut']} semnale)\n"
            f"• 🛑 Stop-Loss: {tehnic['stop_loss']} (-{tehnic['risc_proc']}%) | "
            f"🎯 TP1: {tehnic['tp1']} (+{tehnic['profit1_proc']}%) | 🚀 TP2: {tehnic['tp2']}\n"
            f"• Poziționare (risc 2% din {capital_referinta}): {nr_actiuni} acțiuni (~{round(nr_actiuni * tehnic['pret'], 2)} {moneda})"
        )

    upside_str = ""
    if isinstance(target_price, (int, float)) and isinstance(pret_ref, (int, float)) and pret_ref > 0:
        upside = ((target_price - pret_ref) / pret_ref) * 100
        upside_str = f" (țintă analiști: {target_price:.2f} {moneda}, potențial {upside:+.1f}%)"

    stiri = _extrage_stiri_yahoo(tk)
    bloc_stiri = "\n".join(stiri) if stiri else "• Nu au fost găsite știri recente în fluxul Yahoo Finance."
    fund_detalii = "\n  • ".join(argumente_fund) if argumente_fund else "Date financiare parțiale (specific firmelor mici/de nișă)"
    pe_str = f"{forward_pe:.1f}" if isinstance(forward_pe, (int, float)) else (f"{trailing_pe:.1f}" if isinstance(trailing_pe, (int, float)) else "N/A")

    return (
        f"🏢 EVALUARE COMPLETĂ COMPANIE: {nume_companie} ({sim})\n"
        f"Sector: {sector} | Industrie: {industrie} | Capitalizare: {mcap_str}\n\n"
        f"1️⃣ SĂNĂTATE FUNDAMENTALĂ (Scor: {scor_fund}/100):\n"
        f"• P/E: {pe_str} | Consens analiști: {recomandare_analisti}{upside_str}\n"
        f"  • {fund_detalii}\n\n"
        f"2️⃣ ANALIZĂ TEHNICĂ & NIVELURI DE INTRARE:\n"
        f"{bloc_tehnic}\n\n"
        f"3️⃣ ULTIMELE ȘTIRI:\n"
        f"{bloc_stiri}\n\n"
        f"Sintetizează datele de mai sus și explică clar dacă mișcarea actuală este cauzată de știri trecătoare pe o companie sănătoasă fundamental sau de o problemă reală a afacerii."
    )
