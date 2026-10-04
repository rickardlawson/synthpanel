"""«Ti på gata»: de største grupperingene i et utvalg, vist som personer.

Metode
------
1. Agentene i utvalget kodes på tvers av verdihierarkiet – hygienefaktorer
   (alder, husholdning, utdanning, status, økonomi, bolig, bosted),
   motivasjonsfaktorer (medier, netthandel, parti) og verdifaktorer (ESS-verdier).
2. Vektet k-means (k ≤ 10) deler utvalget i grupperinger. Til sammen dekker de
   hele utvalget, og hver har en andel av det.
3. Hver gruppering vises som én *ekte* syntetisk agent – den som ligger nærmest
   midten av gruppen – så profilen alltid er en sammenhengende person, ikke et
   gjennomsnitt. Navn og portrett velges ut fra kjønn, alder og bakgrunn.

Samme utvalg gir alltid samme personas (frø fra filtrene).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from synthpanel.personas import names, portraits

K = 10
MAX_SAMPLE = 6000
MIN_PER_CLUSTER = 15

ORD3 = {"lav": 0.0, "middels": 0.5, "høy": 1.0}
TIER_WEIGHT = {"hygiene": 1.0, "motivasjon": 0.7, "verdi": 0.8}

# (kolonne, lag, type, vekt). type: cat = kategorisk, num = tall (skalert), ord = ordnet, bin = ja/nei
FEATURES = [
    ("alder", "hygiene", "num", 1.4), ("kjonn", "hygiene", "cat", 0.8), ("sentralitet", "hygiene", "ord", 1.0),
    ("utdanning", "hygiene", "cat", 1.0), ("arbeidsstatus", "hygiene", "cat", 1.2), ("husholdning", "hygiene", "cat", 1.2),
    ("inntektsdesil", "hygiene", "ord", 1.0), ("eierstatus", "hygiene", "cat", 0.7), ("boligtype", "hygiene", "cat", 0.7),
    ("bakgrunn", "hygiene", "cat", 0.8),
    ("parti_2025", "motivasjon", "cat", 1.0), ("politisk_interesse", "motivasjon", "cat", 0.5),
    *[(c, "motivasjon", "bin", 0.35) for c in (
        "daglig_facebook", "daglig_instagram", "daglig_snapchat", "daglig_tiktok", "daglig_youtube", "daglig_linkedin",
        "daglig_nrktv", "daglig_netflix", "daglig_tv2play", "daglig_viaplay", "daglig_disney",
        "netthandel_dagligvarer", "netthandel_klaer", "netthandel_reiser", "netthandel_takeaway", "netthandel_kosmetikk")],
    *[(c, "verdi", "ord", 0.8) for c in (
        "verdi_apenhet", "verdi_trygghet", "verdi_selvhevdelse", "verdi_fellesskap",
        "tillit", "risikovilje", "klimabekymring", "religiositet")],
    ("politisk_sted", "verdi", "ord", 0.8),
    ("treningsfrekvens", "motivasjon", "cat", 0.5),
    *[(c, "motivasjon", "bin", 0.3) for c in (
        "trening_lop", "trening_styrke", "trening_langrenn", "trening_sykkel", "trening_svom", "trening_yoga",
        "trening_fotball", "trening_golf", "friluft_fottur", "friluft_skitur", "friluft_alpint", "friluft_fiske",
        "friluft_jakt", "friluft_baer", "friluft_overnatting", "friluft_bat")],
]
ORDINALS = {"sentralitet": lambda s: (s.astype(int) - 1) / 5, "inntektsdesil": lambda s: (s.astype(int) - 1) / 9,
            "politisk_sted": lambda s: s.map({"venstre": 0.0, "sentrum": 0.5, "høyre": 1.0})}


def _encode(df: pd.DataFrame, skip: set[str]) -> np.ndarray:
    blocks = []
    for col, tier, kind, w in FEATURES:
        if col not in df.columns or col in skip:
            continue
        w = w * TIER_WEIGHT[tier]
        s = df[col]
        if kind == "num":
            blocks.append(((s - 18) / 25).to_numpy(float)[:, None] * w)
        elif kind == "ord":
            v = ORDINALS[col](s) if col in ORDINALS else s.map(ORD3)
            blocks.append(v.fillna(0.5).to_numpy(float)[:, None] * w)
        elif kind == "bin":
            blocks.append((s == "ja").to_numpy(float)[:, None] * w)
        else:
            blocks.append(pd.get_dummies(s).to_numpy(float) * w / np.sqrt(2))
    return np.hstack(blocks) if blocks else np.zeros((len(df), 1))


def _kmeans(X: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """Vektet k-means med k-means++-start. Returnerer klyngenummer per rad."""
    n = len(X)
    centers = [X[rng.choice(n, p=w / w.sum())]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = w * d2
        centers.append(X[rng.choice(n, p=p / p.sum())])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    C = np.array(centers)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        newC = np.array([np.average(X[lab == j], axis=0, weights=w[lab == j]) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1), C


# ---------------------------------------------------------------------------
# Tekst
# ---------------------------------------------------------------------------
STATUS = {"sysselsatt": "i jobb", "student": "student", "pensjonist": "pensjonist", "aap_ufor": "på AAP eller uføretrygd",
          "arbeidsledig": "arbeidsledig", "tiltak": "på arbeidsmarkedstiltak", "annet": "utenfor arbeidslivet"}
EDU = {"grunnskole": "grunnskole", "videregaende": "videregående", "fagskole": "fagskole",
       "uh_kort": "bachelor eller annen kort høyere utdanning", "uh_lang": "master eller mer", "uoppgitt": "uoppgitt utdanning"}
TYPE = {"enebolig": "enebolig", "tomannsbolig": "tomannsbolig", "rekkehus_smahus": "rekkehus",
        "blokk": "leilighet i blokk", "annen": "annen boligtype"}
PARTY = {"A": "Ap", "FRP": "FrP", "H": "Høyre", "SP": "Sp", "SV": "SV", "RØDT": "Rødt", "MDG": "MDG", "KRF": "KrF",
         "V": "Venstre", "ANDRE": "et mindre parti"}
MEDIA = {"daglig_facebook": "Facebook", "daglig_instagram": "Instagram", "daglig_snapchat": "Snapchat",
         "daglig_tiktok": "TikTok", "daglig_youtube": "YouTube", "daglig_linkedin": "LinkedIn", "daglig_nrktv": "NRK TV",
         "daglig_netflix": "Netflix", "daglig_tv2play": "TV 2 Play", "daglig_viaplay": "Viaplay", "daglig_disney": "Disney+"}
SHOP = {"netthandel_dagligvarer": "dagligvarer", "netthandel_klaer": "klær", "netthandel_reiser": "reiser",
        "netthandel_takeaway": "take-away", "netthandel_kosmetikk": "kosmetikk"}
TRAIN = {"trening_lop": "løping", "trening_styrke": "styrketrening", "trening_langrenn": "langrenn",
         "trening_sykkel": "sykling", "trening_svom": "svømming", "trening_yoga": "yoga", "trening_fotball": "fotball",
         "trening_golf": "golf"}
OUTDOOR = {"friluft_fottur": "lange fotturer", "friluft_skitur": "skiturer", "friluft_alpint": "alpint",
           "friluft_fiske": "fisking", "friluft_jakt": "jakt", "friluft_baer": "bær- og sopptur",
           "friluft_overnatting": "overnatting ute", "friluft_bat": "båtturer"}
FREQ = {"ukentlig": "Trener hver uke", "av_og_til": "Trener av og til", "sjelden": "Trener sjelden eller aldri"}
VALUES = {"verdi_apenhet": ("åpen for nye ting og opplevelser", "foretrekker det kjente framfor det nye"),
          "verdi_trygghet": ("setter trygghet, orden og tradisjon høyt", "bryr seg lite om tradisjoner og regler"),
          "verdi_selvhevdelse": ("er opptatt av å lykkes og bli lagt merke til", "er lite opptatt av status og suksess"),
          "verdi_fellesskap": ("er opptatt av å ta vare på andre og naturen", "er mer opptatt av eget liv enn av fellesskapet"),
          "risikovilje": ("liker å ta sjanser", "unngår risiko"),
          "tillit": ("har høy tillit til Storting, politi og rettsvesen", "har lav tillit til politikere og institusjoner"),
          "klimabekymring": ("er bekymret for klimaendringene", "er lite bekymret for klimaendringene"),
          "religiositet": ("er religiøs", "er ikke religiøs")}


def joinog(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " og " + xs[-1]


def household(a: pd.Series) -> str:
    h, age = a["husholdning"], a["alder"]
    return {
        "aleneboende": "Bor alene",
        "par_uten_barn": "Bor med partner, ingen barn hjemme",
        "par_smaa_barn": "Bor med partner og barn under skolealder",
        "par_store_barn": "Bor med partner og barn i skolealder",
        "enslig_smaa_barn": "Alene med små barn",
        "enslig_store_barn": "Alene med barn i skolealder",
        "voksne_barn": "Bor hjemme hos foreldrene" if age < 30 else "Bor sammen med voksne barn",
        "flerfamilie": "Bor i en flerfamiliehusholdning",
    }.get(h, "Bor i en annen type husholdning")


def housing(a: pd.Series) -> str:
    t = TYPE.get(a["boligtype"], "bolig")
    return {"selveier": f"Eier {t}", "andelseier": f"Bor i borettslag ({t})", "leier": f"Leier {t}"}.get(a["eierstatus"], t)


def income(a: pd.Series) -> str:
    d = int(a["inntektsdesil"])
    lvl = "lav" if d <= 3 else "middels" if d <= 7 else "høy"
    return f"{lvl} husholdningsinntekt (desil {d})" + (", under lavinntektsgrensen" if a["lavinntekt"] == "ja" else "")


def title(a: pd.Series) -> str:
    sex, st, h, age = a["kjonn"], a["arbeidsstatus"], a["husholdning"], a["alder"]
    parent = "mor" if sex == "kvinne" else "far"
    if st == "student":
        noun = "Student"
    elif st == "pensjonist":
        noun = "Pensjonist" if h != "aleneboende" else "Pensjonist som bor alene"
    elif st == "aap_ufor":
        noun = "Uføretrygdet" if age >= 45 else "På AAP"
    elif st == "arbeidsledig":
        noun = "Jobbsøker"
    elif h in ("par_smaa_barn", "enslig_smaa_barn"):
        noun = ("Alenemor" if sex == "kvinne" else "Alenefar") if h.startswith("enslig") else f"Småbarns{parent}"
    elif h in ("par_store_barn", "enslig_store_barn"):
        noun = f"Skolebarns{parent}" if h.startswith("par") else ("Alenemor" if sex == "kvinne" else "Alenefar")
    elif h == "voksne_barn" and age < 30:
        noun = "Ung voksen som bor hjemme"
    elif h == "aleneboende":
        noun = "Single i jobb" if age < 45 else "Yrkesaktiv som bor alene"
    elif h == "par_uten_barn":
        noun = "Ungt par" if age < 35 else "Etablert, barna har flyttet ut" if age >= 50 else "Samboer uten barn"
    else:
        noun = "Yrkesaktiv"
    place = {"01": "i storbyområde", "02": "i storbyområde", "03": "i regionsenter", "04": "i småby"}.get(a["sentralitet"], "på bygda")
    return f"{noun} {place}"


def values_text(a: pd.Series) -> list[str]:
    hi = [VALUES[c][0] for c in VALUES if c in a and a[c] == "høy"]
    lo = [VALUES[c][1] for c in VALUES if c in a and a[c] == "lav"]
    return hi[:3] + lo[:2]


def archetype_rows(a: pd.Series) -> list[list[str]]:
    """Arketypelaget (tolkning) – navn og kort forklaring fra configs/archetypes.yaml."""
    if a.get("arketype") in (None, "ukjent"):
        return []
    from synthpanel import config
    c = config.load("archetypes")
    ar, ar2 = c["arketyper"].get(a["arketype"], {}), c["arketyper"].get(a.get("arketype_2"), {})
    low = lambda t: (t[:1].lower() + t[1:]).rstrip(".")  # noqa: E731
    rows = [["Arketype", f"{ar.get('navn')} – {low(ar.get('kort', ''))}" + (f" (med trekk av {ar2['navn'].lower()})" if ar2 else "")]]
    for col, key, label in (("verdikart", ("verdikart", "felt"), "Verdikart"), ("samfunnsrolle", ("samfunnsroller",), "Samfunnsrolle"),
                            ("resiliens", ("resiliens",), "Kriseresiliens")):
        node = c
        for k in key:
            node = node[k]
        m = node.get(a.get(col), {})
        if m:
            rows.append([label, f"{m['navn']} – {low(m['kort'])}"])
    return rows


def profile(a: pd.Series, labels: dict) -> dict:
    """Personens egenskaper gruppert etter verdihierarkiet."""
    lab = lambda col: labels.get(col, {}).get(str(a[col]), str(a[col]))  # noqa: E731
    media = [v for c, v in MEDIA.items() if a.get(c) == "ja"]
    shop = [v for c, v in SHOP.items() if a.get(c) == "ja"]
    train = [v for c, v in TRAIN.items() if a.get(c) == "ja"]
    outd = [v for c, v in OUTDOOR.items() if a.get(c) == "ja"]
    vote = a.get("parti_2025")
    vote_t = ("Stemte ikke ved valget i 2025" if vote == "stemte_ikke" else "Har ikke stemmerett" if vote == "ikke_stemmerett"
              else f"Stemte {PARTY.get(vote, vote)} i 2025")
    return {
        "hygiene": [
            ["Bosted", f"{a['kommune_navn']}, {a['fylke_navn'].split(' - ')[0]}"],
            ["Husholdning", household(a)],
            ["Utdanning", EDU.get(a["utdanning"], a["utdanning"]).capitalize()],
            ["Hovedstatus", STATUS.get(a["arbeidsstatus"], a["arbeidsstatus"]).capitalize()],
            ["Bolig", housing(a)],
            ["Økonomi", income(a).capitalize()],
            ["Bakgrunn", lab("innvkat") if a["innvkat"] != "ovrige" else "Norsk"],
        ],
        "motivasjon": [
            ["Daglige medier", joinog(media) if media else "Ingen av de store tjenestene daglig"],
            ["Handler på nett", joinog(shop).capitalize() if shop else "Lite netthandel"],
            *([["Trening", FREQ.get(a["treningsfrekvens"], "") + (": " + joinog(train) if train else "")]]
              if "treningsfrekvens" in a else []),
            *([["Friluftsliv", joinog(outd).capitalize() if outd else "Lite friluftsliv siste år"]]
              if "friluft_fottur" in a else []),
            ["Politikk", vote_t + (f" · står nærmest {PARTY.get(a['partisympati'], a['partisympati'])}"
                                   if a.get("partisympati") and a.get("partisympati") != vote else "")],
            ["Politisk interesse", "Høy" if a.get("politisk_interesse") == "høy" else "Lav"],
        ],
        "verdi": [
            [labels["_titles"].get(c, c), lab(c)] for c in
            ("verdi_apenhet", "verdi_trygghet", "verdi_selvhevdelse", "verdi_fellesskap", "risikovilje", "tillit",
             "klimabekymring", "religiositet", "politisk_sted") if c in a
        ],
        "arketype": archetype_rows(a),
    }


def story(name: str, a: pd.Series) -> str:
    first = name.split()[0]
    st = a["arbeidsstatus"]
    work = {"sysselsatt": "er i jobb", "student": "studerer", "pensjonist": "er pensjonist",
            "aap_ufor": "mottar AAP eller uføretrygd", "arbeidsledig": "er arbeidsledig og søker jobb",
            "tiltak": "er på arbeidsmarkedstiltak"}.get(st, "står utenfor arbeidslivet")
    s = [f"{first} er {a['alder']} år, bor i {a['kommune_navn']} og {work}.",
         f"{household(a)}, og {housing(a).lower()}."]
    media = [v for c, v in MEDIA.items() if a.get(c) == "ja"]
    if media:
        s.append(f"Er innom {joinog(media[:4])} hver dag.")
    train = [v for c, v in TRAIN.items() if a.get(c) == "ja"]
    outd = [v for c, v in OUTDOOR.items() if a.get(c) == "ja"]
    if a.get("treningsfrekvens") == "ukentlig" and train:
        s.append(f"Trener hver uke, mest {joinog(train[:2])}.")
    elif a.get("treningsfrekvens") == "sjelden":
        s.append("Trener sjelden.")
    if outd:
        s.append(f"Har vært på {joinog(outd[:3])} det siste året." if not {"fisking", "jakt"} & set(outd[:3])
                 else f"Friluftsliv: {joinog(outd[:3])}.")
    v = values_text(a)
    if v:
        s.append(f"{first} {joinog(v[:3])}.")
    vote = a.get("parti_2025")
    if vote in PARTY:
        s.append(f"Stemte {PARTY[vote]} ved stortingsvalget.")
    elif vote == "stemte_ikke":
        s.append("Stemte ikke ved stortingsvalget.")
    return " ".join(s)


def brief(name: str, a: pd.Series, prof: dict, share: float, persons: int) -> str:
    """Kort persona-beskrivelse til bruk som grunnlag når personaen skal svare på spørsmål (L4)."""
    lines = [f"Du er {name}, {a['alder']} år ({'kvinne' if a['kjonn'] == 'kvinne' else 'mann'}).",
             f"Du representerer en gruppe på om lag {persons:,} voksne i Norge ({share:.0%} av utvalget).".replace(",", " ")]
    for tier, title_ in (("hygiene", "Livssituasjon"), ("motivasjon", "Interesser og vaner"), ("verdi", "Verdier"),
                         ("arketype", "Personlighet og samfunnsrolle (tolkning)")):
        if prof.get(tier):
            lines.append(f"{title_}: " + "; ".join(f"{k.lower()}: {v}" for k, v in prof[tier]) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def distinctive(cl: pd.DataFrame, seg: pd.DataFrame, skip: set[str], labels: dict, tier_of: dict, n: int = 4) -> list[dict]:
    """Det som skiller grupperingen fra resten av utvalget (lift mot utvalget)."""
    out = []
    wc, ws = cl["vekt"], seg["vekt"]
    for col in [f[0] for f in FEATURES] + ["arketype", "verdikart", "samfunnsrolle", "resiliens"]:
        if col in skip or col == "alder" or col not in cl.columns:
            continue
        a = cl.groupby(col)["vekt"].sum() / wc.sum()
        b = seg.groupby(col)["vekt"].sum() / ws.sum()
        for val, share in a.items():
            if (col in MEDIA or col in SHOP or col in TRAIN or col in OUTDOOR) and val != "ja":
                continue
            lift = share / b.get(val, np.nan)
            if share >= 0.45 and lift >= 1.35:
                if col in MEDIA:
                    lab = f"Bruker {MEDIA[col]} daglig"
                elif col in SHOP:
                    lab = f"Handler {SHOP[col]} på nett"
                elif col in TRAIN:
                    lab = f"Driver med {TRAIN[col]}"
                elif col in OUTDOOR:
                    lab = f"Friluftsliv: {OUTDOOR[col]}"
                elif col == "treningsfrekvens":
                    lab = FREQ.get(str(val), str(val))
                elif col == "utdanning":
                    lab = {"grunnskole": "Grunnskole som høyeste utdanning", "videregaende": "Videregående som høyeste utdanning",
                           "fagskole": "Fagskoleutdannet", "uh_kort": "Kort høyere utdanning",
                           "uh_lang": "Lang høyere utdanning"}.get(str(val), str(val))
                elif col in ("arketype", "verdikart", "samfunnsrolle", "resiliens"):
                    if val == "ukjent":
                        continue
                    lab = labels.get(col, {}).get(str(val), str(val))
                elif col == "kjonn":
                    lab = "Kvinne" if val == "kvinne" else "Mann"
                else:
                    lab = labels.get(col, {}).get(str(val), str(val))
                    phrase = labels.get("_phrases", {}).get(col)
                    if phrase:
                        lab = phrase.replace("{}", lab if col == "parti_2025" else lab.lower())
                        lab = lab[0].upper() + lab[1:]
                ctx = ("daglig" if col in MEDIA else "netthandel" if col in SHOP else "trening" if col in TRAIN
                       else "friluftsliv" if col in OUTDOOR else labels["_titles"].get(col, col))
                out.append({"dim": col, "verdi": str(val), "tekst": lab, "kontekst": ctx, "andel": float(share),
                            "lift": float(lift), "lag": tier_of.get(col)})
    out.sort(key=lambda r: -(r["lift"] * r["andel"]))
    seen, res = set(), []
    for r in out:
        if r["dim"] not in seen:
            seen.add(r["dim"]); res.append(r)
    return res[:n]


def build(seg: pd.DataFrame, filtered: set[str], labels: dict, tier_of: dict, key: str, k: int = K) -> dict:
    n = len(seg)
    k = max(1, min(k, n // MIN_PER_CLUSTER))
    if n < 30:
        return {"personas": [], "agenter": n, "melding": "For få syntetiske personer i utvalget til å lage personas."}
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    sample = seg.sample(n=min(n, MAX_SAMPLE), random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    skip = {c for c in filtered if c in sample.columns and sample[c].nunique() <= 1} | filtered
    X = _encode(sample, skip)
    w = sample["vekt"].to_numpy(float)
    lab, C = _kmeans(X, w, k, rng)
    total = w.sum()
    seg_persons = float(seg["vekt"].sum())

    taken_img, taken_names, out = set(), set(), []
    for j in range(k):
        m = lab == j
        if not m.any():
            continue
        idx = np.where(m)[0]
        # Representanten skal ha gruppens vanligste kjønn, hovedstatus og husholdning –
        # ellers kan nærmeste-midtpunkt-regelen gi skjev kjønnsbalanse i galleriet.
        cl_ = sample.iloc[idx]
        cand = np.ones(len(idx), bool)
        for col in ("kjonn", "arbeidsstatus", "husholdning"):
            mode = cl_.groupby(col)["vekt"].sum().idxmax()
            nxt = cand & (cl_[col] == mode).to_numpy()
            if nxt.any():
                cand = nxt
        ci = idx[cand]
        rep = sample.iloc[ci[((X[ci] - C[j]) ** 2).sum(1).argmin()]]
        share = float(w[m].sum() / total)
        out.append((share, j, rep, sample[m]))
    out.sort(key=lambda t: -t[0])

    personas = []
    for rank, (share, j, a, cl) in enumerate(out, 1):
        pid = a["agent_id"]
        a = a.copy()
        a["kommune_navn"] = str(a["kommune_navn"]).split(" - ")[0]
        img = portraits.pick(a["bakgrunn"], a["kjonn"], int(a["alder"]), pid, taken_img)
        origin = img["opphav"] if img else "Norwegian"
        if a["bakgrunn"] == "norsk":
            origin = "Norwegian"
        name = names.name_for(a["kjonn"], 2026 - int(a["alder"]), origin, pid, taken_names)
        prof = profile(a, labels)
        persons = round(share * seg_persons)
        personas.append({
            "rang": rank, "id": pid, "navn": name, "alder": int(a["alder"]), "kjonn": a["kjonn"],
            "tittel": title(a), "sted": a["kommune_navn"], "portrett": img["fil"] if img else None,
            "andel": share, "personer": persons, "agenter_i_gruppen": len(cl),
            "historie": story(name, a), "profil": prof,
            "arketype": a.get("arketype") if a.get("arketype") != "ukjent" else None,
            "kjennetegn": distinctive(cl, sample, skip, labels, tier_of),
            "brief": brief(name, a, prof, share, persons),
        })
    return {"personas": personas, "agenter": n, "k": len(personas)}
