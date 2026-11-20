import json
import os
from datetime import datetime
import requests
import yfinance as yf
from custom_tools import _incarca_watchlist, _calculeaza_metrici_actiune, _extrage_stiri_yahoo

CANAL_NTFY = os.environ.get("NTFY_CANAL", "").strip()
FISIER_CACHE_ALERTE = "alerte_trimise_cache.json"


def _incarca_cache() -> dict:
    if os.path.exists(FISIER_CACHE_ALERTE):
        try:
            with open(FISIER_CACHE_ALERTE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _salveaza_cache(date: dict):
    with open(FISIER_CACHE_ALERTE, "w", encoding="utf-8") as f:
        json.dump(date, f, indent=2)


def _trimite_ntfy(titlu: str, mesaj: str, tag: str = "chart_with_upwards_trend"):
    if not CANAL_NTFY:
        print(f"[DRY-RUN fără NTFY_CANAL] {titlu}\n{mesaj}\n")
        return
    requests.post(
        f"https://ntfy.sh/{CANAL_NTFY}",
        data=mesaj.encode("utf-8"),
        headers={"Title": titlu, "Priority": "high", "Tags": tag},
        timeout=10,
    )


def _verifica_fundamental_dip(simbol: str) -> tuple[bool, str]:
    """Verifică dacă o companie solidă fundamental are creștere de venituri și profitabilitate bună."""
    try:
        tk = yf.Ticker(simbol)
        info = tk.info or {}
        rev = info.get("revenueGrowth") or 0
        margin = info.get("profitMargins") or 0
        target = info.get("targetMeanPrice")
        stiri = _extrage_stiri_yahoo(tk)
        prima_stire = stiri[0] if stiri else "Fără știri recente"
        if rev > 0.12 and margin > 0.10:
            detalii = f"Venituri +{rev*100:.0f}%, Marjă {margin*100:.0f}%"
            if target:
                detalii += f", Țintă analiști: {target}"
            return True, f"{detalii} | {prima_stire}"
    except Exception:
        pass
    return False, ""


def main():
    azi = datetime.utcnow().strftime("%Y-%m-%d")
    cache = _incarca_cache()
    # Curățăm intrările din zilele trecute
    cache = {k: v for k, v in cache.items() if k.startswith(azi)}

    watchlist = _incarca_watchlist()
    print(f"Scanez {len(watchlist)} simboluri pe GitHub Actions ({azi})...")

    for sim in watchlist:
        rez = _calculeaza_metrici_actiune(sim)
        if not rez:
            continue

        # 1. Alertă Setup Tehnic Puternic (Scor >= 70)
        cheie_tech = f"{azi}_TECH_{sim}"
        if rez["scor"] >= 70 and not cache.get(cheie_tech):
            motive = "; ".join(rez["motive"][:2])
            mesaj = (
                f"Pret: {rez['pret']} ({rez['variatie_zi']:+.2f}% azi) | Win-rate 2Y: {rez['win_rate']}%\n"
                f"Stop-Loss: {rez['stop_loss']} (-{rez['risc_proc']}%)\n"
                f"Take-Profit 1: {rez['tp1']} (+{rez['profit1_proc']}%)\n"
                f"Take-Profit 2: {rez['tp2']}\n"
                f"Motiv: {motive}"
            )
            _trimite_ntfy(f"JARVIS Setup ({rez['scor']}/100): {sim}", mesaj)
            cache[cheie_tech] = True

        # 2. Alertă "Buy the Dip" (companie sănătoasă care scade brusc > 4.5% într-o zi)
        cheie_dip = f"{azi}_DIP_{sim}"
        if rez["variatie_zi"] <= -4.5 and not cache.get(cheie_dip):
            e_solida, info_fund = _verifica_fundamental_dip(sim)
            if e_solida:
                mesaj = (
                    f"Scadere azi: {rez['variatie_zi']}% (Pret: {rez['pret']})\n"
                    f"Companie solida: {info_fund}\n"
                    f"Stop-Loss sugerat: {rez['stop_loss']} | TP1: {rez['tp1']}"
                )
                _trimite_ntfy(f"JARVIS Dip Alert ({rez['variatie_zi']}%): {sim}", mesaj, tag="rotating_light")
                cache[cheie_dip] = True

    _salveaza_cache(cache)
    print("Scanare finalizată.")


if __name__ == "__main__":
    main()
