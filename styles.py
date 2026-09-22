"""Résolution des styles (héritage basedOn, docDefaults) et de la numérotation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from lxml import etree

from .ooxml import on_off, qn, wattr


@dataclass(slots=True, frozen=True)
class RunFormat:
    """Mise en forme effective d'un run (None = non défini à ce niveau)."""
    size: float | None = None       # points
    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    caps: bool | None = None
    color: str | None = None
    font: str | None = None
    vanish: bool | None = None      # texte masqué

    def merged(self, over: "RunFormat") -> "RunFormat":
        """Superpose `over` (plus spécifique) sur self."""
        return RunFormat(*(o if o is not None else s for s, o in zip(_astuple(self), _astuple(over))))


def _astuple(f: RunFormat) -> tuple:
    return (f.size, f.bold, f.italic, f.underline, f.caps, f.color, f.font, f.vanish)


def parse_rpr(rpr: etree._Element | None) -> RunFormat:
    if rpr is None:
        return RunFormat()
    size = None
    sz = rpr.find(qn("w:sz"))
    if sz is not None:
        try:
            size = int(wattr(sz, "val")) / 2.0
        except (TypeError, ValueError):
            size = None
    u = rpr.find(qn("w:u"))
    underline = None if u is None else wattr(u, "val", "single") != "none"
    color_el = rpr.find(qn("w:color"))
    color = wattr(color_el, "val") if color_el is not None else None
    fonts = rpr.find(qn("w:rFonts"))
    font = (wattr(fonts, "ascii") or wattr(fonts, "hAnsi")) if fonts is not None else None
    caps = on_off(rpr.find(qn("w:caps")))
    if caps is None:
        caps = on_off(rpr.find(qn("w:smallCaps")))
    return RunFormat(size=size, bold=on_off(rpr.find(qn("w:b"))), italic=on_off(rpr.find(qn("w:i"))),
                     underline=underline, caps=caps, color=color, font=font,
                     vanish=on_off(rpr.find(qn("w:vanish"))))


@dataclass(slots=True)
class ResolvedStyle:
    style_id: str
    name: str
    names_chain: tuple[str, ...]          # nom du style puis ancêtres (minuscules)
    ids_chain: tuple[str, ...]
    outline_lvl: int | None               # 0..8 (9 = corps de texte -> None)
    num_id: str | None
    ilvl: int | None
    run: RunFormat
    jc: str | None
    is_default: bool = False


class StyleResolver:
    def __init__(self, styles_root: etree._Element | None):
        self._raw: dict[str, etree._Element] = {}
        self._char_cache: dict[str, RunFormat] = {}
        self._para_cache: dict[str, ResolvedStyle] = {}
        self.default_para_style: str | None = None
        self.doc_defaults = RunFormat(size=10.0)  # défaut Word si docDefaults absent
        if styles_root is None:
            return
        dd = styles_root.find(f"{qn('w:docDefaults')}/{qn('w:rPrDefault')}/{qn('w:rPr')}")
        self.doc_defaults = self.doc_defaults.merged(parse_rpr(dd))
        for st in styles_root.iter(qn("w:style")):
            sid = wattr(st, "styleId")
            if not sid:
                continue
            self._raw[sid] = st
            if wattr(st, "type") == "paragraph" and wattr(st, "default") in ("1", "true"):
                self.default_para_style = sid

    def name_of(self, sid: str | None) -> str:
        st = self._raw.get(sid or "")
        if st is None:
            return sid or ""
        n = st.find(qn("w:name"))
        return (wattr(n, "val") or sid) if n is not None else sid

    def paragraph(self, sid: str | None) -> ResolvedStyle:
        sid = sid or self.default_para_style or ""
        cached = self._para_cache.get(sid)
        if cached is not None:
            return cached
        chain: list[etree._Element] = []
        seen, cur = set(), sid
        while cur and cur in self._raw and cur not in seen and len(chain) < 32:  # anti-cycle
            seen.add(cur)
            chain.append(self._raw[cur])
            based = self._raw[cur].find(qn("w:basedOn"))
            cur = wattr(based, "val") if based is not None else None
        run = self.doc_defaults
        outline = num_id = ilvl = jc = None
        for st in reversed(chain):  # du plus générique au plus spécifique
            run = run.merged(parse_rpr(st.find(qn("w:rPr"))))
            ppr = st.find(qn("w:pPr"))
            if ppr is None:
                continue
            ol = ppr.find(qn("w:outlineLvl"))
            if ol is not None:
                outline = _int(wattr(ol, "val"))
            numpr = ppr.find(qn("w:numPr"))
            if numpr is not None:
                nid = numpr.find(qn("w:numId"))
                lv = numpr.find(qn("w:ilvl"))
                if nid is not None:
                    num_id = wattr(nid, "val")
                if lv is not None:
                    ilvl = _int(wattr(lv, "val"))
            j = ppr.find(qn("w:jc"))
            if j is not None:
                jc = wattr(j, "val")
        if outline is not None and not (0 <= outline <= 8):
            outline = None
        names = tuple(self.name_of(wattr(st, "styleId")).strip().lower() for st in chain) or ((sid or "").lower(),)
        rs = ResolvedStyle(style_id=sid, name=self.name_of(sid), names_chain=names,
                           ids_chain=tuple((wattr(st, "styleId") or "").lower() for st in chain),
                           outline_lvl=outline, num_id=None if num_id == "0" else num_id,
                           ilvl=ilvl, run=run, jc=jc, is_default=sid == self.default_para_style)
        self._para_cache[sid] = rs
        return rs

    def character(self, sid: str | None) -> RunFormat:
        if not sid:
            return RunFormat()
        cached = self._char_cache.get(sid)
        if cached is not None:
            return cached
        chain, seen, cur = [], set(), sid
        while cur and cur in self._raw and cur not in seen and len(chain) < 32:
            seen.add(cur)
            chain.append(self._raw[cur])
            b = self._raw[cur].find(qn("w:basedOn"))
            cur = wattr(b, "val") if b is not None else None
        fmt = RunFormat()
        for st in reversed(chain):
            fmt = fmt.merged(parse_rpr(st.find(qn("w:rPr"))))
        self._char_cache[sid] = fmt
        return fmt


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


# ============================================================ numérotation
@dataclass(slots=True)
class Level:
    num_fmt: str = "decimal"
    lvl_text: str = "%1."
    start: int = 1
    p_style: str | None = None
    is_lgl: bool = False

    @property
    def is_bullet(self) -> bool:
        return self.num_fmt in ("bullet", "none")


_BULLET_MAP = {"\uf0b7": "•", "\uf0a7": "▪", "\uf0d8": "➢", "\uf076": "❖", "\uf0fc": "✓",
               "\uf0a8": "□", "\uf06f": "o", "\uf02d": "-", "\uf0e0": "→", "o": "◦", "": "•"}


def normalize_bullet(ch: str) -> str:
    ch = ch.strip()
    return _BULLET_MAP.get(ch, ch if ch and not ("\uf000" <= ch[:1] <= "\uf0ff") else "•")


class NumberingResolver:
    """Calcule les marqueurs de liste (« 1.2.a) ») à partir de numbering.xml."""

    def __init__(self, root: etree._Element | None):
        self.abstract: dict[str, dict[int, Level]] = {}
        self.num_to_abs: dict[str, str] = {}
        self.overrides: dict[str, dict[int, Level | int]] = {}
        self.counters: dict[str, list[int]] = {}
        self._started_nums: set[str] = set()
        if root is None:
            return
        for an in root.iter(qn("w:abstractNum")):
            aid = wattr(an, "abstractNumId")
            self.abstract[aid] = {(_int(wattr(l, "ilvl")) or 0): self._parse_lvl(l) for l in an.iter(qn("w:lvl"))}
        link_map = {}
        for an in root.iter(qn("w:abstractNum")):  # numStyleLink -> styleLink
            sl = an.find(qn("w:numStyleLink"))
            if sl is not None:
                link_map[wattr(an, "abstractNumId")] = wattr(sl, "val")
        for num in root.iter(qn("w:num")):
            nid = wattr(num, "numId")
            a = num.find(qn("w:abstractNumId"))
            if a is not None:
                self.num_to_abs[nid] = wattr(a, "val")
            ov: dict[int, Level | int] = {}
            for lo in num.iter(qn("w:lvlOverride")):
                il = _int(wattr(lo, "ilvl")) or 0
                so = lo.find(qn("w:startOverride"))
                lvl = lo.find(qn("w:lvl"))
                if lvl is not None:
                    ov[il] = self._parse_lvl(lvl)
                elif so is not None:
                    ov[il] = _int(wattr(so, "val")) or 1
            if ov:
                self.overrides[nid] = ov

    @staticmethod
    def _parse_lvl(l: etree._Element) -> Level:
        def val(tag, default=None):
            e = l.find(qn(tag))
            return wattr(e, "val", default) if e is not None else default
        return Level(num_fmt=val("w:numFmt", "decimal"), lvl_text=val("w:lvlText", "") or "",
                     start=_int(val("w:start", "1")) or 0, p_style=val("w:pStyle"),
                     is_lgl=l.find(qn("w:isLgl")) is not None)

    def level(self, num_id: str, ilvl: int) -> Level | None:
        ov = self.overrides.get(num_id, {}).get(ilvl)
        if isinstance(ov, Level):
            return ov
        levels = self.abstract.get(self.num_to_abs.get(num_id, ""), {})
        return levels.get(ilvl)

    def heading_styles(self) -> set[str]:
        """Styles liés à une numérotation multi-niveaux (souvent des titres)."""
        return {l.p_style for lv in self.abstract.values() for l in lv.values() if l.p_style}

    def next_marker(self, num_id: str, ilvl: int) -> tuple[str, Level | None]:
        lvl = self.level(num_id, ilvl)
        if lvl is None:
            return "", None
        key = self.num_to_abs.get(num_id, num_id)
        cnt = self.counters.setdefault(key, [0] * 9)
        # startOverride : redémarrage à la première utilisation de cette instance w:num
        if num_id not in self._started_nums:
            self._started_nums.add(num_id)
            for il, ov in self.overrides.get(num_id, {}).items():
                if isinstance(ov, int) and 0 <= il < 9:
                    cnt[il] = ov - 1
                    for d in range(il + 1, 9):
                        cnt[d] = 0
        if not (0 <= ilvl < 9):
            return "", lvl
        if cnt[ilvl] == 0:
            cnt[ilvl] = lvl.start
        else:
            cnt[ilvl] += 1
        for d in range(ilvl + 1, 9):
            cnt[d] = 0
        if lvl.is_bullet:
            return (normalize_bullet(lvl.lvl_text) if lvl.num_fmt == "bullet" else ""), lvl

        def repl(m: re.Match) -> str:
            k = int(m.group(1)) - 1
            if not 0 <= k < 9:
                return ""
            lk = self.level(num_id, k) or lvl
            value = cnt[k] if cnt[k] else (lk.start or 1)
            return format_number(value, "decimal" if lvl.is_lgl else lk.num_fmt)
        return re.sub(r"%(\d)", repl, lvl.lvl_text), lvl


def _roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out or "0"


def _letters(n: int) -> str:
    if n <= 0:
        return "0"
    # Word répète la lettre : 27 -> aa
    return chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)


def format_number(n: int, fmt: str) -> str:
    if fmt == "lowerLetter":
        return _letters(n)
    if fmt == "upperLetter":
        return _letters(n).upper()
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n).upper()
    if fmt == "decimalZero":
        return f"{n:02d}"
    if fmt == "none":
        return ""
    return str(n)
