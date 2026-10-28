"""Synthetic insurance policy generator for the SFT + LoRA experiment.

Every policy, exclusion, exception and claim is generated from templates, so the
correct answer is known exactly. Some wordings are held back and only appear in
the test set, so the test measures whether the model learned the task rather
than memorised sentences.
"""
import json
import random
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
INSURERS = ["Northwind Mutual", "Harborline Assurance", "Crescent General", "Bluepeak Insurance Co.",
            "Meridian Underwriters", "Oakridge Indemnity", "Silverstone Casualty", "Tidewater Insurance Group"]
COMPANIES = ["Larkspur Bakery LLC", "Vantage Cold Storage Inc.", "Copperline Logistics", "Juniper Dental Group",
             "Brightwater Textiles", "Kestrel Analytics Ltd.", "Redfern Auto Parts", "Mosaic Coffee Roasters",
             "Halcyon Printworks", "Quarry Lane Furniture", "Northgate Pharmacy", "Pinecrest Veterinary Clinic",
             "Atlas Freight Partners", "Sunfield Organics", "Ironbridge Machining", "Cobalt Data Systems"]
PEOPLE = ["Maria Okafor", "Daniel Reyes", "Priya Natarajan", "Tom Brennan", "Aisha Rahman", "Lucas Meyer",
          "Grace Liu", "Samuel Adeyemi", "Elena Petrova", "Rohan Mehta", "Hannah Schultz", "Carlos Duarte"]

POLICY_TYPES = {
    "commercial_property": {"title": ["Commercial Property Policy", "Business Property Coverage Form", "Commercial Property Insurance Contract"],
                            "insured": COMPANIES, "limit": (250_000, 5_000_000), "deductible": [1_000, 2_500, 5_000, 10_000, 25_000]},
    "homeowners": {"title": ["Homeowners Policy", "Homeowners Insurance Policy (HO-3)", "Residential Dwelling Policy"],
                   "insured": PEOPLE, "limit": (150_000, 1_500_000), "deductible": [500, 1_000, 2_000, 2_500, 5_000]},
    "commercial_auto": {"title": ["Commercial Auto Policy", "Business Auto Coverage Form", "Fleet Vehicle Insurance Policy"],
                        "insured": COMPANIES, "limit": (100_000, 2_000_000), "deductible": [500, 1_000, 2_500, 5_000]},
    "cyber_liability": {"title": ["Cyber Liability Policy", "Cyber and Data Breach Coverage Form", "Network Security and Privacy Policy"],
                        "insured": COMPANIES, "limit": (250_000, 10_000_000), "deductible": [5_000, 10_000, 25_000, 50_000]},
}

# Each excludable peril: several surface wordings (index 0 is held out for the test set),
# exceptions (wording variants, events that meet them, events that don't), and plain events.
PERILS = {
    # ---------------- property / homeowners ----------------
    "flood": {"types": ["commercial_property", "homeowners"],
              "wordings": ["inundation by surface water, overflow of any body of water, or tidal surge",
                           "flood, surface water, waves, or overflow of any body of water",
                           "water damage caused by flooding from rivers, lakes, or storm surge",
                           "rising water from any natural source, whether or not driven by wind"],
              "exceptions": [(["unless the water escapes from a plumbing, heating, or fire sprinkler system inside the insured building",
                               "However, this exclusion does not apply to water that leaks from a burst pipe or sprinkler system within the premises"],
                              ["a pipe burst on the second floor and water flooded the stockroom",
                               "a fire sprinkler line ruptured overnight and soaked the ground floor"],
                              ["the nearby river overflowed after three days of rain and entered the building",
                               "storm surge pushed seawater through the loading dock doors"])],
              "plain": ["the nearby creek overflowed its banks and flooded the ground floor"]},
    "earthquake": {"types": ["commercial_property", "homeowners"],
                   "wordings": ["any movement of the earth's crust, including tremors and aftershocks",
                                "earthquake, landslide, mudflow, or other earth movement",
                                "earth movement of any kind, including sinkhole collapse and subsidence",
                                "seismic activity or ground shifting from any cause"],
                   "exceptions": [(["unless fire breaks out as a direct result, in which case we cover the fire damage",
                                    "However, if earth movement results in fire, we will pay for the loss caused by that fire"],
                                   ["an earthquake ruptured a gas line and the resulting fire destroyed the kitchen",
                                    "tremors knocked over a space heater, which started a fire in the office"],
                                   ["an earthquake cracked the foundation and the east wall partially collapsed",
                                    "ground subsidence caused the rear slab to split"])],
                   "plain": ["an earthquake cracked the foundation and damaged the interior walls"]},
    "mold": {"types": ["commercial_property", "homeowners"],
             "wordings": ["the presence, growth, or spread of fungi, wet rot, or dry rot",
                          "mold, mildew, fungus, or bacteria of any kind",
                          "fungal growth or microbial contamination",
                          "damage from mold or spores, however caused"],
             "exceptions": [(["unless the mold results from a covered water loss that is reported within 14 days",
                              "This exclusion does not apply to mold caused by a covered water loss that you report to us within 14 days"],
                             ["mold spread after a covered pipe leak that was reported to the insurer within 3 days",
                              "fungal growth followed a covered sprinkler leak reported to the insurer the next day"],
                             ["mold spread behind the drywall after a slow leak that went unreported for two months",
                              "mildew built up over years in a poorly ventilated basement"])],
             "plain": ["black mold was found throughout the basement storage area"]},
    "wear_and_tear": {"types": ["commercial_property", "homeowners", "commercial_auto"],
                      "wordings": ["gradual deterioration, rust, corrosion, or latent defect",
                                   "wear and tear, marring, or deterioration",
                                   "mechanical breakdown or gradual wearing out of parts",
                                   "deterioration over time or inherent vice"],
                      "exceptions": [],
                      "plain": ["the roof membrane wore out after twenty years and began to leak",
                                "worn brake pads failed and damaged the rotors"]},
    "war": {"types": ["commercial_property", "homeowners", "commercial_auto", "cyber_liability"],
            "wordings": ["hostile or warlike action in time of peace or war, including insurrection or rebellion",
                         "war, including undeclared or civil war",
                         "acts of war, invasion, or military action by any government or sovereign power",
                         "warlike operations, whether or not war is declared"],
            "exceptions": [],
            "plain": ["the premises were damaged during armed conflict in the region",
                      "a state-sponsored military operation disabled the insured's systems"]},
    "power_failure": {"types": ["commercial_property"],
                      "wordings": ["interruption of power or other utility service supplied from off the premises",
                                   "power failure or utility service interruption",
                                   "loss caused by the failure of power, communication, or water supply services",
                                   "any outage of off-site electricity or utilities"],
                      "exceptions": [(["unless the interruption results in a covered fire on the premises, and then we pay only for the fire damage",
                                       "However, if the utility interruption causes a covered fire at the described premises, we will pay for that fire damage"],
                                      ["a power surge after the grid outage ignited a fire in the electrical room",
                                       "when power was restored, an arc in the main panel started a fire"],
                                      ["a three-day grid outage spoiled all refrigerated inventory",
                                       "the utility cut power for 48 hours and production halted"])],
                      "plain": ["a regional grid outage spoiled the refrigerated inventory"]},
    "freezing": {"types": ["homeowners"],
                 "wordings": ["freezing of a plumbing system while the dwelling is vacant or unheated",
                              "freezing of plumbing, heating, or air conditioning systems",
                              "frozen pipes and the water damage that follows",
                              "damage from ice forming inside pipes or appliances"],
                 "exceptions": [(["unless you used reasonable care to maintain heat in the building",
                                  "This exclusion does not apply if you maintained heat in the dwelling"],
                                 ["a pipe froze and burst during a cold snap even though the heating was running at 68 degrees",
                                  "pipes froze in an exterior wall while the thermostat was kept at 65 degrees"],
                                 ["pipes froze while the house was left unheated for the winter",
                                  "the owner shut off the heat during a month-long trip and the pipes burst"])],
                 "plain": ["a pipe froze and burst in the crawl space"]},
    # ---------------- auto ----------------
    "racing": {"types": ["commercial_auto"],
               "wordings": ["use of a covered auto in any organized speed contest or demolition event",
                            "racing, speed testing, or stunting",
                            "participation in any prearranged race or timed driving competition",
                            "operation of the vehicle in speed trials or track events"],
               "exceptions": [],
               "plain": ["the van crashed during an amateur track day at the speedway"]},
    "intentional_damage": {"types": ["commercial_auto", "homeowners"],
                           "wordings": ["damage expected or intended by the insured",
                                        "intentional acts of any insured",
                                        "loss deliberately caused by you or anyone acting on your behalf",
                                        "willful or malicious acts committed by an insured"],
                           "exceptions": [],
                           "plain": ["an employee deliberately rammed the truck into a gate after an argument"]},
    "unlicensed_driver": {"types": ["commercial_auto"],
                          "wordings": ["operation of a covered auto by any person without a valid driver's license",
                                       "use by an unlicensed driver",
                                       "loss occurring while the auto is driven by someone not legally licensed to drive it",
                                       "any accident involving a driver whose license is suspended, revoked, or expired"],
                          "exceptions": [(["unless the vehicle was stolen and driven by the thief",
                                           "However, this exclusion does not apply when the auto was being driven by a person who had stolen it"],
                                          ["the truck was stolen from the depot and the thief, who had no license, crashed it",
                                           "a joyrider with no license stole the delivery van and hit a barrier"],
                                          ["a new hire with an expired license drove the box truck into a pole",
                                           "the owner's nephew, who never had a license, borrowed the van and crashed it"])],
                          "plain": ["a driver with a suspended license rear-ended another vehicle"]},
    "flood_auto": {"types": ["commercial_auto"],
                   "wordings": ["water entering the vehicle while driven through standing water",
                                "driving into flood water or high water",
                                "damage from deliberately operating the vehicle in flooded roads",
                                "submersion caused by driving through water deeper than the wheel hubs"],
                   "exceptions": [(["unless the vehicle was parked and lawfully unattended when the water rose",
                                    "This exclusion does not apply to a vehicle that was parked and unattended when flooding occurred"],
                                   ["the van was parked overnight in the lot when flash flooding submerged it",
                                    "flood water rose around the parked and locked truck while the driver was home"],
                                   ["the driver tried to cross a flooded underpass and the engine stalled",
                                    "the truck was driven through a flooded road and water entered the engine"])],
                   "plain": ["the driver steered into a flooded underpass and the engine was ruined"]},
    # ---------------- cyber ----------------
    "prior_known_breach": {"types": ["cyber_liability"],
                           "wordings": ["any incident known to a senior officer before the inception date",
                                        "prior or pending incidents known before the policy began",
                                        "any breach that you were aware of, or should reasonably have known about, before inception",
                                        "circumstances reported under any prior policy"],
                           "exceptions": [],
                           "plain": ["the breach traced back to an intrusion the CTO had flagged two months before the policy started"]},
    "unencrypted_device": {"types": ["cyber_liability"],
                           "wordings": ["loss of data stored on any portable device that is not encrypted",
                                        "theft of unencrypted laptops, phones, or storage media",
                                        "data exposure from any unencrypted mobile device",
                                        "breaches arising from lost or stolen devices without full-disk encryption"],
                           "exceptions": [(["unless the device was stolen in a burglary with forced entry reported to police within 48 hours",
                                            "However, we will cover loss from a stolen unencrypted device if it was taken in a forced-entry burglary reported to police within 48 hours"],
                                           ["an unencrypted laptop was stolen in an office break-in that was reported to police the same night",
                                            "burglars forced the back door and took an unencrypted file server; police were called within an hour"],
                                           ["an employee left an unencrypted laptop on a train and it was never recovered",
                                            "an unencrypted phone went missing at a conference and the loss was reported a week later"])],
                           "plain": ["an unencrypted laptop with customer records was lost in a taxi"]},
    "infrastructure_failure": {"types": ["cyber_liability"],
                               "wordings": ["failure of any power, internet, or telecommunications infrastructure not under your control",
                                            "outages of third-party utilities or internet service providers",
                                            "interruption of external networks, satellites, or power grids",
                                            "failure of infrastructure operated by a public utility"],
                               "exceptions": [(["unless the outage is caused by a cyber attack aimed directly at your systems",
                                                "This exclusion does not apply if the outage results from a targeted cyber attack on your network"],
                                               ["attackers targeted the company's own firewall, knocking its internet connection offline for two days",
                                                "a denial-of-service attack aimed at the insured's servers took the store offline"],
                                               ["the regional internet provider suffered a fiber cut and the store was offline for two days",
                                                "a citywide power outage took the data center offline"])],
                               "plain": ["a regional internet provider outage took the online store offline"]},
}

# Events that are always covered (no exclusion applies), per policy type.
COVERED_EVENTS = {
    "commercial_property": [("fire", "a fire started in the warehouse and destroyed racking and stock"),
                            ("windstorm", "a windstorm tore off part of the roof"),
                            ("vandalism", "vandals smashed the storefront windows and damaged displays"),
                            ("theft", "burglars broke in and stole equipment from the workshop")],
    "homeowners": [("fire", "a kitchen fire spread to the dining room"),
                   ("windstorm", "high winds brought a tree down onto the garage"),
                   ("theft", "thieves broke a window and stole electronics"),
                   ("hail", "hail damaged the roof shingles and skylights")],
    "commercial_auto": [("collision", "a delivery truck was rear-ended at a traffic light"),
                        ("hail", "hail dented the roofs of three parked vans"),
                        ("theft", "a company pickup was stolen from a client's lot"),
                        ("vandalism", "vandals slashed the tires and broke the windshield of a box truck")],
    "cyber_liability": [("ransomware", "ransomware encrypted the company's file servers"),
                        ("phishing", "an employee fell for a phishing email and attackers copied the customer database"),
                        ("data_breach", "attackers exploited a web application flaw and stole payment records"),
                        ("business_email_compromise", "criminals hijacked an executive's mailbox and redirected a supplier payment")],
}

ALL_PERILS = sorted(set(PERILS) | {p for v in COVERED_EVENTS.values() for p, _ in v})
DECISIONS = ["covered", "excluded", "covered_by_exception", "outside_policy_period"]

SYSTEM_PROMPT = (
    "You are an insurance policy analyst. Read the policy and the claim, then return ONLY a JSON object, "
    "with no other text, using exactly this schema:\n"
    "{\"policy_number\": string, \"policy_type\": one of [\"commercial_property\", \"homeowners\", \"commercial_auto\", \"cyber_liability\"], "
    "\"insured\": string, \"effective_date\": \"YYYY-MM-DD\", \"expiration_date\": \"YYYY-MM-DD\", "
    "\"coverage_limit_usd\": integer, \"deductible_usd\": integer, "
    "\"exclusions\": [{\"peril\": string, \"has_exception\": boolean}], "
    "\"claim\": {\"peril\": string, \"decision\": one of [\"covered\", \"excluded\", \"covered_by_exception\", \"outside_policy_period\"], \"payable_usd\": integer}}\n"
    "Rules: list only exclusions that are still in force after any endorsements. "
    "Peril labels must come from this list: " + ", ".join(ALL_PERILS) + ". "
    "payable_usd = min(claim amount, coverage limit) minus the deductible, never below 0, and 0 when the claim is not covered."
)

# ---------------------------------------------------------------------------
# Formatting helpers (several styles; some reserved for the test set)
# ---------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def fmt_date(d, style):
    return [lambda: f"{MONTHS[d.month-1]} {d.day}, {d.year}",
            lambda: f"{d.month:02d}/{d.day:02d}/{d.year}",
            lambda: d.isoformat(),
            lambda: f"{d.day} {MONTHS[d.month-1]} {d.year}"][style]()


def fmt_money(x, style):
    return [lambda: f"${x:,}",
            lambda: f"USD {x:,}",
            lambda: f"{x:,} US dollars",
            lambda: (f"${x/1_000_000:g} million" if x >= 1_000_000 and x % 100_000 == 0 else f"${x:,}")][style]()


FILLER = [
    "Throughout this policy, \"you\" and \"your\" refer to the Named Insured shown in the Declarations, and \"we\", \"us\" and \"our\" refer to the company providing this insurance.",
    "Other words and phrases that appear in quotation marks have special meaning. Refer to the Definitions section.",
    "You must notify us promptly of any loss and cooperate with us in the investigation or settlement of the claim.",
    "We will not pay for loss unless you have complied with all of the terms of this policy, including the duties after loss.",
    "This policy may be cancelled by you at any time by written notice. We may cancel by giving you 30 days' written notice.",
    "If there is other insurance covering the same loss, we will pay only our share of the loss.",
    "No one may bring a legal action against us unless there has been full compliance with all terms of this policy.",
    "Premium shown in the Declarations was computed based on rates in effect at the time the policy was issued.",
]

EXC_INTRO = ["We will not pay for loss or damage caused directly or indirectly by any of the following:",
             "This insurance does not apply to:",
             "The following exclusions apply. We do not cover loss resulting from:",
             "Coverage under this policy is excluded for:"]

LETTERS = "ABCDEFGH"


def make_example(rng, split):
    """Return (prompt_text, target_dict). split is 'train' or 'test'."""
    test = split == "test"
    ptype = rng.choice(list(POLICY_TYPES))
    pt = POLICY_TYPES[ptype]
    insurer = rng.choice(INSURERS)
    insured = rng.choice(pt["insured"])
    pnum = f"{rng.choice(['CP', 'HO', 'CA', 'CY', 'PL', 'BX'])}-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}"
    eff = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    exp = eff.replace(year=eff.year + 1) if not (eff.month == 2 and eff.day == 29) else eff + timedelta(days=365)
    limit = rng.randrange(pt["limit"][0], pt["limit"][1] + 1, 50_000)
    deductible = rng.choice(pt["deductible"])

    # Held-out styles: date style 3, money style 3 and wording index 0 appear only in test.
    date_style = rng.choice([0, 1, 2, 3]) if test else rng.choice([0, 1, 2])
    money_style = rng.choice([0, 1, 2, 3]) if test else rng.choice([0, 1, 2])

    pool = [p for p, v in PERILS.items() if ptype in v["types"]]
    n_exc = min(len(pool), rng.randint(2, 4))
    chosen = rng.sample(pool, n_exc)
    exclusions = []
    for peril in chosen:
        info = PERILS[peril]
        widx = (rng.choice([0, 1, 2, 3]) if test else rng.choice([1, 2, 3]))
        has_exc = bool(info["exceptions"]) and rng.random() < 0.6
        exc = None
        if has_exc:
            texts, met, unmet = info["exceptions"][0]
            # the first "met"/"unmet" event of each exception is only used in the test set
            exc = {"text": rng.choice(texts) if test else texts[0],
                   "met": met if test else met[1:], "unmet": unmet if test else unmet[1:]}
        exclusions.append({"peril": peril, "wording": info["wordings"][widx], "exc": exc})

    # Optional endorsement that deletes one exclusion.
    deleted = None
    if len(exclusions) >= 2 and rng.random() < 0.25:
        deleted = rng.randrange(len(exclusions))

    # ---- Claim ----
    r = rng.random()
    claim_date = eff + timedelta(days=rng.randint(10, 340))
    in_force = [i for i in range(len(exclusions)) if i != deleted]
    with_exc = [i for i in in_force if exclusions[i]["exc"]]
    no_exc = [i for i in in_force if not exclusions[i]["exc"]]
    if r < 0.08:
        claim_date = (exp + timedelta(days=rng.randint(5, 120))) if rng.random() < 0.5 else (eff - timedelta(days=rng.randint(5, 120)))
        peril, event = rng.choice(COVERED_EVENTS[ptype])
        decision = "outside_policy_period"
    elif r < 0.40 and with_exc:
        i = rng.choice(with_exc); peril = exclusions[i]["peril"]
        event = rng.choice(exclusions[i]["exc"]["met"]); decision = "covered_by_exception"
    elif r < 0.58 and with_exc:
        i = rng.choice(with_exc); peril = exclusions[i]["peril"]
        event = rng.choice(exclusions[i]["exc"]["unmet"]); decision = "excluded"
    elif r < 0.70 and no_exc:
        i = rng.choice(no_exc); peril = exclusions[i]["peril"]
        event = rng.choice(PERILS[peril]["plain"]); decision = "excluded"
    elif r < 0.80 and deleted is not None:
        peril = exclusions[deleted]["peril"]
        event = rng.choice(PERILS[peril]["plain"]); decision = "covered"
    else:
        peril, event = rng.choice(COVERED_EVENTS[ptype]); decision = "covered"

    claim_amount = rng.randrange(5_000, int(limit * 1.3), 500)
    payable = max(0, min(claim_amount, limit) - deductible) if decision in ("covered", "covered_by_exception") else 0

    # ---- Render the document ----
    title = rng.choice(pt["title"])
    lines = [f"{insurer.upper()}", f"{title}", "",
             "DECLARATIONS",
             f"Policy Number: {pnum}",
             f"Named Insured: {insured}",
             f"Policy Period: {fmt_date(eff, date_style)} to {fmt_date(exp, date_style)}, 12:01 a.m. standard time",
             f"Limit of Insurance: {fmt_money(limit, money_style)} per occurrence",
             f"Deductible: {fmt_money(deductible, money_style)} per occurrence", ""]
    rng.shuffle(FILLER_COPY := FILLER[:])
    lines += ["SECTION I - GENERAL PROVISIONS", FILLER_COPY[0], FILLER_COPY[1], "",
              "SECTION II - EXCLUSIONS", rng.choice(EXC_INTRO)]
    for k, e in enumerate(exclusions):
        s = f"{LETTERS[k]}. {e['wording'][0].upper()}{e['wording'][1:]}"
        if e["exc"]:
            t = e["exc"]["text"]
            s += (", " + t + ".") if t.startswith("unless") else (". " + t + ".")
        else:
            s += "."
        lines.append(s)
    lines += ["", "SECTION III - CONDITIONS", FILLER_COPY[2], FILLER_COPY[3]]
    if deleted is not None:
        verb = rng.choice(["is deleted in its entirety", "is removed from this policy", "no longer applies"])
        lines += ["", "ENDORSEMENTS",
                  f"Endorsement No. {rng.randint(1, 9)}: Exclusion {LETTERS[deleted]} in Section II {verb}, effective from the start of the policy period."]
    lines += ["", "CLAIM NOTICE",
              f"On {fmt_date(claim_date, date_style)}, {event}. The insured is claiming {fmt_money(claim_amount, money_style)}."]
    doc = "\n".join(lines)

    target = {
        "policy_number": pnum,
        "policy_type": ptype,
        "insured": insured,
        "effective_date": eff.isoformat(),
        "expiration_date": exp.isoformat(),
        "coverage_limit_usd": limit,
        "deductible_usd": deductible,
        "exclusions": [{"peril": e["peril"], "has_exception": bool(e["exc"])} for i, e in enumerate(exclusions) if i != deleted],
        "claim": {"peril": peril, "decision": decision, "payable_usd": payable},
    }
    return doc, target


def build_dataset(n_train=800, n_test=150, seed=42):
    rng = random.Random(seed)
    train = [make_example(rng, "train") for _ in range(n_train)]
    test = [make_example(rng, "test") for _ in range(n_test)]
    return train, test


def to_messages(doc, target=None):
    """SageMaker SFT 'messages' format (the same file also works in SageMaker serverless model customization)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": doc}]
    if target is not None:
        msgs.append({"role": "assistant", "content": json.dumps(target)})
    return {"messages": msgs}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
TOP_FIELDS = ["policy_number", "policy_type", "insured", "effective_date", "expiration_date", "coverage_limit_usd", "deductible_usd"]


def parse_json(text):
    """Return (obj or None, strict_ok). strict_ok means the whole reply was a single JSON object."""
    t = text.strip()
    try:
        return json.loads(t), True
    except Exception:
        pass
    # lenient: strip code fences / take the first {...} block
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            try:
                return json.loads(p), False
            except Exception:
                continue
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1]), False
        except Exception:
            return None, False
    return None, False


def schema_ok(o):
    try:
        if not isinstance(o, dict) or set(o) != set(TOP_FIELDS + ["exclusions", "claim"]):
            return False
        if not all(isinstance(o[f], str) for f in ["policy_number", "policy_type", "insured", "effective_date", "expiration_date"]):
            return False
        if not (isinstance(o["coverage_limit_usd"], int) and isinstance(o["deductible_usd"], int)):
            return False
        if not isinstance(o["exclusions"], list) or not all(isinstance(x, dict) and set(x) == {"peril", "has_exception"} and isinstance(x["has_exception"], bool) for x in o["exclusions"]):
            return False
        c = o["claim"]
        return isinstance(c, dict) and set(c) == {"peril", "decision", "payable_usd"} and c["decision"] in DECISIONS and isinstance(c["payable_usd"], int)
    except Exception:
        return False


def score(pred_text, gold):
    o, strict = parse_json(pred_text)
    res = {"valid_json": o is not None, "strict_json": strict and o is not None, "schema_ok": schema_ok(o) if o is not None else False}
    o = o if isinstance(o, dict) else {}
    for f in TOP_FIELDS:
        res[f] = str(o.get(f, "")).strip().lower() == str(gold[f]).strip().lower()
    res["fields_acc"] = sum(res[f] for f in TOP_FIELDS) / len(TOP_FIELDS)
    gold_exc = {(e["peril"], e["has_exception"]) for e in gold["exclusions"]}
    try:
        pred_exc = {(str(e.get("peril")), bool(e.get("has_exception"))) for e in o.get("exclusions", []) if isinstance(e, dict)}
    except Exception:
        pred_exc = set()
    tp = len(gold_exc & pred_exc)
    prec = tp / len(pred_exc) if pred_exc else 0.0
    rec = tp / len(gold_exc) if gold_exc else (1.0 if not pred_exc else 0.0)
    res["exclusions_f1"] = (2 * prec * rec / (prec + rec)) if (prec + rec) else (1.0 if not gold_exc and not pred_exc else 0.0)
    c = o.get("claim", {}) if isinstance(o.get("claim", {}), dict) else {}
    res["claim_peril"] = c.get("peril") == gold["claim"]["peril"]
    res["decision"] = c.get("decision") == gold["claim"]["decision"]
    res["payable"] = c.get("payable_usd") == gold["claim"]["payable_usd"]
    res["all_correct"] = bool(res["schema_ok"] and res["fields_acc"] == 1 and res["exclusions_f1"] == 1 and res["claim_peril"] and res["decision"] and res["payable"])
    return res


def summarize(scores):
    keys = ["valid_json", "strict_json", "schema_ok", "fields_acc", "exclusions_f1", "claim_peril", "decision", "payable", "all_correct"]
    n = len(scores)
    return {k: round(100 * sum(float(s[k]) for s in scores) / n, 1) for k in keys}


if __name__ == "__main__":
    tr, te = build_dataset(10, 5)
    d, t = te[0]
    print(d); print(json.dumps(t, indent=2))
    print(score(json.dumps(t), t))
