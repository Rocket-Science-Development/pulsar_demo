"""LP-02 — Canadian identifiers at scale with hard negatives, and leak counting in scripted agent conversations.

A seeded generator writes FR/EN sentences that carry one synthetic identifier (Luhn-valid fake SIN, RAMQ number,
postal code, NANP phone) in the formats people actually type, with and without a context word, and 30 % hard
negatives: case numbers, amounts, dates, order and product codes, non-Luhn numbers, invalid RAMQ dates. Every value
is generated: none belongs to a real person. Ground truth is the span the generator inserted.

Then scripted conversations: the user gives some identifiers, the agent output repeats some of them (not a leak) and
adds identifiers of someone else (leaks); PiiSession must count the leaks exactly.
"""

import random
import time
from collections import Counter, defaultdict

from ._llm_common import DRY, wilson

CRITERIA = {"min_precision": (">=", 0.95), "min_recall": (">=", 0.95), "leak_count_exact_share": ("==", 1.0),
            "sentences_per_s": (">=", 1000)}
PARAMS = {"sentences": 100_000, "hard_negative_share": 0.30, "conversations": 5_000, "seed": 7, "dry": DRY}
TYPES = ("sin", "ramq", "postal_code", "phone")
POSTAL_FIRST = "ABCEGHJKLMNPRSTVXY"
POSTAL_OTHER = "ABCEGHJKLMNPRSTVWXYZ"


def luhn_complete(prefix):
    for d in range(10):
        n = prefix + str(d)
        total = 0
        for i, c in enumerate(reversed(n)):
            x = int(c)
            if i % 2 == 1:
                x *= 2
                x = x - 9 if x > 9 else x
            total += x
        if total % 10 == 0:
            return n
    raise AssertionError


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    # ---- values (format variant recorded)
    def sin(self):
        r = self.r
        digits = luhn_complete(r.choice("1234567 9".replace(" ", "")) + "".join(str(r.randrange(10)) for _ in range(7)))
        fmt = r.choices(["space", "dash", "plain"], [0.5, 0.3, 0.2])[0]
        sep = {"space": " ", "dash": "-", "plain": ""}[fmt]
        return sep.join([digits[:3], digits[3:6], digits[6:]]), fmt

    def ramq(self):
        r = self.r
        letters = "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4))
        yy, mm, dd = r.randrange(100), r.randrange(1, 13) + (50 if r.random() < 0.5 else 0), r.randrange(1, 29)
        nn = r.randrange(10, 100)
        compact = f"{letters}{yy:02d}{mm:02d}{dd:02d}{nn:02d}"
        fmt = r.choices(["spaced", "compact", "lower", "dashed"], [0.55, 0.25, 0.1, 0.1])[0]
        if fmt == "spaced":
            v = f"{compact[:4]} {compact[4:8]} {compact[8:]}"
        elif fmt == "compact":
            v = compact
        elif fmt == "lower":
            v = f"{compact[:4].lower()} {compact[4:8]} {compact[8:]}"
        else:
            v = f"{compact[:4]}-{compact[4:8]}-{compact[8:]}"
        return v, fmt

    def postal(self):
        r = self.r
        v = r.choice(POSTAL_FIRST) + str(r.randrange(10)) + r.choice(POSTAL_OTHER) + " " + str(r.randrange(10)) + \
            r.choice(POSTAL_OTHER) + str(r.randrange(10))
        fmt = r.choices(["space", "nospace", "lower"], [0.6, 0.25, 0.15])[0]
        if fmt == "nospace":
            v = v.replace(" ", "")
        elif fmt == "lower":
            v = v.lower()
        return v, fmt

    def phone(self):
        r = self.r
        area = r.choice(["514", "438", "450", "418", "581", "819", "873", "613", "416", "647", "604", "403", "902", "800", "888"])
        ex = str(r.randrange(2, 10)) + f"{r.randrange(100):02d}"
        ln = f"{r.randrange(10000):04d}"
        fmt = r.choices(["dash", "dot", "space", "paren", "plus1", "plain", "1dash", "ext"], [0.25, 0.1, 0.15, 0.2, 0.1, 0.08, 0.07, 0.05])[0]
        v = {"dash": f"{area}-{ex}-{ln}", "dot": f"{area}.{ex}.{ln}", "space": f"{area} {ex} {ln}",
             "paren": f"({area}) {ex}-{ln}", "plus1": f"+1 {area} {ex} {ln}", "plain": f"{area}{ex}{ln}",
             "1dash": f"1-{area}-{ex}-{ln}", "ext": f"{area}-{ex}-{ln}"}[fmt]
        return v, fmt

    # ---- sentences
    POS = {
        "sin": {"fr": ["Mon NAS est le {v}.", "Voici mon numéro d'assurance sociale : {v}", "{v} — c'est mon NAS, merci de le noter.",
                       "Je vous confirme le {v} pour la demande.", "Inscrivez {v} dans le formulaire svp."],
                "en": ["My SIN is {v}.", "Social insurance number: {v}", "Here it is: {v}, please update my profile.",
                       "Please use {v} for the application.", "sin {v}"]},
        "ramq": {"fr": ["Ma carte d'assurance maladie : {v}", "Numéro RAMQ {v}", "Voici ma carte soleil {v}.",
                        "C'est {v} sur ma carte."],
                 "en": ["My health card number is {v}.", "RAMQ: {v}", "The number on my card is {v}."]},
        "postal_code": {"fr": ["J'habite au 1234 rue des Érables, Montréal {v}.", "Code postal : {v}", "Mon adresse se termine par {v}.",
                               "On est dans le secteur {v}."],
                        "en": ["I live at 55 Maple Street, Toronto ON {v}.", "Postal code {v}", "Ship it to {v} please.",
                               "We moved to {v} last year."]},
        "phone": {"fr": ["Appelez-moi au {v}.", "Mon cell : {v}", "Vous pouvez me joindre au {v} après 17 h.",
                         "{v} c'est mon numéro.", "Laissez un message au {v}{ext}."],
                  "en": ["Call me at {v}.", "My phone is {v}.", "Text {v} when ready.", "{v} is the best number.",
                         "Reach me at {v}{ext}."]},
    }

    def positive(self):
        r = self.r
        t = r.choice(TYPES)
        lang = r.choice(["fr", "en"])
        v, fmt = {"sin": self.sin, "ramq": self.ramq, "postal_code": self.postal, "phone": self.phone}[t]()
        template = r.choice(self.POS[t][lang])
        ext = (" poste 1234" if lang == "fr" else " ext. 1234") if fmt == "ext" else ""
        if "{ext}" not in template and fmt == "ext":
            template = template.replace("{v}", "{v}{ext}")
        pre, post = template.split("{v}")
        post = post.replace("{ext}", ext)
        text = pre + v + post
        return text, [(t, len(pre), len(pre) + len(v))], t, fmt, lang

    def member(self):
        """Membership number outside every NANP phone shape (never 10 or 11 digits, never a real area code)"""
        r = self.r
        return r.choice([
            f"{r.randrange(10**7, 10**8)}",
            f"0{r.randrange(100, 1000)} {r.randrange(1000, 10000)} {r.randrange(1000, 10000)}",
            f"{r.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{r.randrange(10**6, 10**7)}",
        ])

    def nonluhn9(self):
        r = self.r
        while True:
            d = "".join(str(r.randrange(10)) for _ in range(9))
            if luhn_complete(d[:8]) != d:
                return d

    def negative(self):
        """Hard negative: looks like an identifier, is not personal information"""
        r = self.r
        lang = r.choice(["fr", "en"])
        kind = r.choice(["case_luhn", "order_phone", "amount", "date", "nonluhn", "ramq_bad_date", "product_postal",
                         "time_range", "invoice_ramq", "ip", "flight", "isbn", "confirmation_luhn",
                         "member_number", "client_number"])
        d9 = luhn_complete("".join(str(r.randrange(10)) for _ in range(8)))
        if d9[0] in "08":
            d9 = "1" + d9[1:]
            d9 = luhn_complete(d9[:8])
        s9 = f"{d9[:3]} {d9[3:6]} {d9[6:]}"
        p = self.phone()[0]
        texts = {
            "case_luhn": {"fr": f"Votre dossier {s9} est en traitement.", "en": f"Your case {s9} is being processed."},
            "order_phone": {"fr": f"Commande {p.replace('(', '').replace(')', '')} expédiée hier.", "en": f"Order {p.replace('(', '').replace(')', '')} shipped yesterday."},
            "amount": {"fr": f"Le solde est de {r.randrange(100, 999)} {r.randrange(100, 999)} {r.randrange(1000, 9999)} $.",
                       "en": f"The balance is {r.randrange(100, 999)} {r.randrange(100, 999)} {r.randrange(1000, 9999)} $."},
            "date": {"fr": f"Rendez-vous le 2026-{r.randrange(1, 13):02d}-{r.randrange(1, 29):02d} à 14 h 30.",
                     "en": f"Appointment on 2026-{r.randrange(1, 13):02d}-{r.randrange(1, 29):02d} at 2:30 pm."},
            "nonluhn": {"fr": f"Le code de suivi est {self.nonluhn9()}.", "en": f"Tracking code {self.nonluhn9()}."},
            "ramq_bad_date": {"fr": f"Référence interne ABCD {r.randrange(10, 99)}{r.choice([13, 14, 33, 45, 70, 99])} {r.randrange(32, 99)}{r.randrange(10, 99)}",
                              "en": f"Internal code ABCD {r.randrange(10, 99)}{r.choice([13, 14, 33, 45, 70, 99])} {r.randrange(32, 99)}{r.randrange(10, 99)}"},
            "product_postal": {"fr": f"Produit {self.postal()[0].upper()} en rupture de stock.", "en": f"Product {self.postal()[0].upper()} is out of stock."},
            "time_range": {"fr": f"Ouvert de 0900 à 1700, local 2{r.randrange(10, 99)}.", "en": f"Open 0900 to 1700, room 2{r.randrange(10, 99)}."},
            "invoice_ramq": {"fr": f"Facture TREM{r.randrange(10, 99)}0{r.randrange(1, 9)}{r.randrange(10, 28)}{r.randrange(10, 99)} payée.",
                             "en": f"Invoice TREM{r.randrange(10, 99)}0{r.randrange(1, 9)}{r.randrange(10, 28)}{r.randrange(10, 99)} paid."},
            "ip": {"fr": f"Serveur 10.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(256)} redémarré.",
                   "en": f"Server 10.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(256)} restarted."},
            "flight": {"fr": f"Vol AC {r.randrange(100, 999)}, porte B{r.randrange(10, 60)}.", "en": f"Flight AC {r.randrange(100, 999)}, gate B{r.randrange(10, 60)}."},
            "confirmation_luhn": {"fr": f"Code de confirmation : {d9}", "en": f"Confirmation number {d9}"},
            # member numbers that do not look like phones (the first run reused the NANP phone generator here, so
            # any detector would see a phone): 8 digits, 4-4-4 with a leading 0, or a letter and 7 digits
            "member_number": {"fr": f"Numéro de membre {self.member()}", "en": f"Member number {self.member()}"},
            "client_number": {"fr": f"No client {s9}, merci.", "en": f"Customer no. {s9}, thank you."},
            "isbn": {"fr": f"ISBN 978-2-{r.randrange(1000, 9999)}-{r.randrange(1000, 9999)}-{r.randrange(10)}",
                     "en": f"ISBN 978-1-{r.randrange(1000, 9999)}-{r.randrange(1000, 9999)}-{r.randrange(10)}"},
        }
        return texts[kind][lang], [], kind, "-", lang


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


def run(ctx):
    from pulsar_data_collection.privacy import PiiDetector, PiiSession

    n = 5_000 if DRY else PARAMS["sentences"]
    gen = Gen(PARAMS["seed"])
    data = [gen.negative() if gen.r.random() < PARAMS["hard_negative_share"] else gen.positive() for _ in range(n)]
    det = PiiDetector(types=TYPES)
    t0 = time.perf_counter()
    found = [det.scan(text) for text, *_ in data]
    seconds = time.perf_counter() - t0

    tp, fp, fn = Counter(), Counter(), Counter()
    by_fmt = defaultdict(Counter)
    neg_fp = Counter()
    neg_n = Counter()
    for (text, gold, kind, fmt, lang), finds in zip(data, found):
        pred = [(f.type, f.start, f.end) for f in finds]
        for g in gold:
            hit = any(p[0] == g[0] and overlap(p, g) for p in pred)
            (tp if hit else fn)[g[0]] += 1
            by_fmt[(g[0], fmt)]["hit" if hit else "miss"] += 1
            by_fmt[(g[0], f"{fmt}/{lang}")]["hit" if hit else "miss"] += 1
        for p in pred:
            if not any(p[0] == g[0] and overlap(p, g) for g in gold):
                fp[p[0]] += 1
                if not gold:
                    neg_fp[(kind, p[0])] += 1
        if not gold:
            neg_n[kind] += 1

    per_type = {}
    for t in TYPES:
        prec = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None
        rec = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else None
        per_type[t] = {"tp": tp[t], "fp": fp[t], "fn": fn[t], "precision": prec and round(prec, 4), "recall": rec and round(rec, 4),
                       "precision_ci": wilson(tp[t], tp[t] + fp[t]), "recall_ci": wilson(tp[t], tp[t] + fn[t])}
    formats = {f"{t}:{f}": {"n": c["hit"] + c["miss"], "recall": round(c["hit"] / (c["hit"] + c["miss"]), 4)}
               for (t, f), c in sorted(by_fmt.items()) if "/" not in f}
    formats_lang = {f"{t}:{f}": {"n": c["hit"] + c["miss"], "recall": round(c["hit"] / (c["hit"] + c["miss"]), 4)}
                    for (t, f), c in sorted(by_fmt.items()) if "/" in f}
    hard = {k: {"n": neg_n[k], "false_findings": {t: neg_fp[(k, t)] for t in TYPES if neg_fp[(k, t)]}} for k in sorted(neg_n)}

    # ---- leak counting in scripted conversations
    conv_n = 500 if DRY else PARAMS["conversations"]
    exact, total_expected, total_counted, off = 0, 0, 0, []
    lg = Gen(PARAMS["seed"] + 1)
    by_style = defaultdict(Counter)
    t1 = time.perf_counter()
    for i in range(conv_n):
        r = lg.r
        own = [lg.positive() for _ in range(r.randint(1, 3))]
        own_values = [(t[0][t[1][0][1]:t[1][0][2]], t[1][0][0]) for t in own]
        other = [lg.positive() for _ in range(r.randint(0, 2))]
        other_values = [(t[0][t[1][0][1]:t[1][0][2]], t[1][0][0]) for t in other]
        user = " ".join(t[0] for t in own)
        repeated = [v for v, _ in own_values if r.random() < 0.6]
        out_parts = ["Merci, j'ai bien noté." if r.random() < 0.5 else "Thanks, noted."]
        out_parts += [f"Je confirme {v}." for v in repeated]
        style = "dossier" if i % 2 else "neutral"  # half the outputs mention a file ("dossier"), as an agent of files does
        out_parts += [(f"Le dossier voisin indique aussi {v}." if style == "dossier" else f"Autre personne : {v}.")
                      for v, _ in other_values]
        output = " ".join(out_parts)
        # ground truth: leaks = values in output detected-as-PII that the user did not provide; evaluate against the
        # values the generator wrote (independent of the detector)
        expected = len(other_values)
        with PiiSession(det) as session:
            session.scan("input", user)
            counted = len(session.scan("output", output).leaks)
        total_expected += expected
        total_counted += counted
        by_style[style]["n"] += 1
        by_style[style]["expected"] += expected
        by_style[style]["counted"] += counted
        if counted == expected:
            exact += 1
            by_style[style]["exact"] += 1
        else:
            off.append(counted - expected)
    conv_s = time.perf_counter() - t1
    leak = {"conversations": conv_n, "exact": exact, "leaks_expected": total_expected, "leaks_counted": total_counted,
            "over_count_convs": sum(1 for d in off if d > 0), "under_count_convs": sum(1 for d in off if d < 0),
            "conversations_per_s": round(conv_n / conv_s, 1), "by_output_style": {k: dict(v) for k, v in by_style.items()}}

    precs = [v["precision"] for v in per_type.values() if v["precision"] is not None]
    recs = [v["recall"] for v in per_type.values() if v["recall"] is not None]
    metrics = {"sentences": n, "hard_negatives": sum(neg_n.values()), "seconds_scan": round(seconds, 2),
               "sentences_per_s": round(n / seconds, 1), "per_type": per_type, "min_precision": min(precs), "min_recall": min(recs),
               "recall_by_format": formats, "recall_by_format_and_language": formats_lang, "hard_negative_findings": hard,
               "leaks": leak, "leak_count_exact_share": round(exact / conv_n, 4), "cpu_host_note": "lab-gpu-1, 4 vCPU (CPU only)"}

    # ---- limits
    for t, v in per_type.items():
        if v["recall"] is not None and v["recall"] < 0.95:
            weak = {k: f for k, f in formats.items() if k.startswith(t + ":") and f["recall"] < 0.9}
            ctx.limit(f"{t} recall {v['recall']} (< 0.95); weakest formats {weak}",
                      f"tp={v['tp']} fn={v['fn']}, Wilson CI {v['recall_ci']}; generator seed {PARAMS['seed']}",
                      f"Extend the {t} recognizer to the missed formats (see recall_by_format) with tests per format.", "major")
        if v["precision"] is not None and v["precision"] < 0.95:
            src = {k: h["false_findings"].get(t) for k, h in hard.items() if h["false_findings"].get(t)}
            ctx.limit(f"{t} precision {v['precision']} (< 0.95) on hard negatives; false findings by negative kind {src}",
                      f"tp={v['tp']} fp={v['fp']}, Wilson CI {v['precision_ci']}",
                      f"Add the missing business words / validations for {t} (see hard_negative_findings) and a "
                      "negative-context list per type rather than one global BUSINESS_WORDS.", "major")
    if exact < conv_n:
        # Not an independent limit: every miscount is an under-count explained by the 'dossier' suppression (limit
        # below, C4 of the report) and by the missed RAMQ formats (recall limit above, M17); the number of conversations
        # off is set by the 50/50 split of output styles. Kept as a note with its numbers.
        ctx.note(f"Leak count not exact in {conv_n - exact} of {conv_n} scripted conversations (expected {total_expected} "
                 f"leaks, counted {total_counted}; over-counted in {leak['over_count_convs']}, under-counted in "
                 f"{leak['under_count_convs']}; by output style {leak['by_output_style']}): consequence of the "
                 "'dossier' suppression and of the recognizer misses, not a separate limit.")
    dos = leak["by_output_style"].get("dossier", {})
    if dos.get("expected") and dos.get("counted", 0) / dos["expected"] < 0.5:
        neu = leak["by_output_style"].get("neutral", {})
        ctx.limit(f"Leaks are hidden when the output mentions a file: with 'dossier' within 40 characters before the value, "
                  f"{dos.get('counted', 0)} of {dos['expected']} leaked identifiers are counted (neutral wording: "
                  f"{neu.get('counted')} of {neu.get('expected')})",
                  "BUSINESS_WORDS lowers SIN/phone/postal/RAMQ confidence to 0.3 (< min_confidence 0.5) for every role, "
                  "including outputs; scripted conversations, seed 8",
                  "Do not apply business-word suppression to the 'output' role of PiiSession (or use a separate, lower "
                  "threshold for leaks); an agent that manages files says 'dossier' in most answers.", "critical")
    if metrics["sentences_per_s"] < 1000:
        ctx.limit(f"PiiDetector throughput {metrics['sentences_per_s']} sentences/s on CPU (< 1k)",
                  f"{n} sentences in {seconds:.1f} s, single thread", "Compile one combined regex per text pass; skip "
                  "recognizers whose cheap prefilter (digit count) fails.", "minor")
    return metrics
