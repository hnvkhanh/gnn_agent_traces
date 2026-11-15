"""Stage 3: concrete-value extraction and preservation check (CPU only).

The extractor pulls URLs, emails, IBANs, file names, money, dates, times, quoted strings, numbers and capitalized
multi-word names out of the original prompt. A paraphrase passes only if every extracted value is still present.
Names are compared after case and whitespace normalization. Numbers and identifiers are compared exactly.
"""
import re

MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec").split()
_MON = r"(?:%s)\.?" % "|".join(sorted(MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
FILE_EXT = ("txt pdf docx doc xlsx xls csv json md pptx ppt png jpg jpeg gif zip py html htm xml yaml yml log "
            "mp3 mp4 wav odt tsv").split()

PATTERNS = [  # order matters: earlier kinds claim their spans first
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s'\"<>]+?[^\s'\"<>.,;:!?)](?=\.[A-Z][a-z]|[.,;:!?)]*(?:\s|$))")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("file", re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:%s)\b" % "|".join(FILE_EXT), re.I)),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b")),
    ("date", re.compile(r"\b%s\s+\d{1,2}%s(?:,?\s+\d{4})?\b|\b\d{1,2}%s(?:\s+of)?\s+%s(?:,?\s+\d{4})?\b"
                        % (_MON, _ORD, _ORD, _MON), re.I)),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?(?![\w])|\b\d{1,2}\s?[ap]\.?m\.?(?![\w])", re.I)),
    ("money", re.compile(r"[$€£]\s?%s|\b%s\s?(?:USD|EUR|GBP|CHF|dollars?|euros?|pounds?)\b" % (_NUM, _NUM),
                         re.I)),
    ("quoted", re.compile(r"(?<![\w])'([^'\n]{2,}?)'(?![\w])|\"([^\"\n]{2,}?)\"|“([^”\n]{2,}?)”")),
    ("enum", re.compile(r"(?:^|(?<=[\s:]))\d{1,2}[.)](?=\s)")),  # list markers "1." / "2)": not values
    ("number", re.compile(r"(?<![\w.])%s(?![\w])" % _NUM)),
]
NAME = re.compile(r"\b[A-Z][a-z][\w'&-]*(?:\s+(?:of\s+|de\s+|van\s+)?[A-Z][a-z][\w'&-]*)+\b")
STOP = set("""a an the i i'm i'd i've i'll can could would will please hi hey hello also then and or but so if
is are was were be my me you your our we us it its this that these those what when where which who how why do does
did make send pay get find give tell show list check read write update create add delete remove book reserve
invite post say visit summarize go rate cancel schedule share
let let's thanks thank ok okay yes no on in at for to from by with of""".split())


def _mask(text, spans):
    s = list(text)
    for a, b in spans:
        s[a:b] = " " * (b - a)
    return "".join(s)


def extract(text):
    """[(kind, value)] in order of appearance. Overlapping matches go to the earlier pattern kind."""
    taken, found = [], []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            a, b = m.span()
            if kind == "quoted":
                g = next(i for i in range(1, 4) if m.group(i) is not None)
                a, b = m.span(g)
                val = m.group(g).strip()
                found.append((a, kind, val))  # quoted content can contain other entities: do not claim the span
                continue
            if free(a, b):
                taken.append((a, b))
                if kind != "enum":
                    found.append((a, kind, m.group(0).strip()))
    masked = _mask(text, taken)
    for m in NAME.finditer(masked):
        words = m.group(0).split()
        while words and words[0].lower().rstrip(",") in STOP:
            words = words[1:]
        while words and words[-1].lower() in STOP:
            words = words[:-1]
        if len(words) >= 2 and not all(w.lower().rstrip(".") in MONTHS for w in words):
            found.append((m.start(), "name", " ".join(words)))
    return [(k, v) for _, k, v in sorted(found, key=lambda x: x[0])]


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def _num_re(n):
    return re.compile(r"(?<![\w.])%s(?![\d])" % re.escape(n))


def present(kind, value, para):
    if kind == "name":
        return _norm_ws(value).casefold() in _norm_ws(para).casefold()
    if kind == "number":
        return bool(_num_re(value.rstrip(".,")).search(para))
    if kind == "money":
        core = re.search(_NUM, value).group(0)
        return bool(_num_re(core).search(para))   # the amount must be unchanged; currency wording may vary
    if kind == "time":
        core = re.search(r"\d{1,2}(?::\d{2})?", value).group(0)
        mer = re.search(r"([ap])\.?m", value, re.I)
        if not _num_re(core).search(para):
            return False
        return mer is None or bool(re.search(re.escape(core) + r"\s?%s\.?m" % mer.group(1), para, re.I))
    if kind == "date" and re.search(_MON, value, re.I) and not re.match(r"\d{4}-", value):
        mon = re.search(_MON, value, re.I).group(0).rstrip(".").casefold()[:3]   # month is a name: case-insensitive
        nums = re.findall(r"\d+", value)
        p = para.casefold()
        return mon in p and all(_num_re(n).search(para) for n in nums)
    return _norm_ws(value) in _norm_ws(para)   # identifiers, ISO/numeric dates, quoted strings: exact


def check(original, paraphrase, entities=None):
    """(ok, reasons) where reasons lists 'missing:<kind>:<value>' for every value not preserved."""
    ents = extract(original) if entities is None else entities
    missing = ["missing:%s:%s" % (k, v) for k, v in ents if not present(k, v, paraphrase)]
    return not missing, missing
