"""Post-process a docx produced by the markdown importer into a submission-grade
document.

The importer used by the local editor writes headings as *direct formatting*
(bold + font size + outlineLvl) and emits an empty styles part.  That renders
acceptably but leaves the document without named styles, without page setup,
and without headers/footers.  This script repairs those things in place:

1. injects a minimal ``styles.xml`` with doc defaults (body font + size) and
   named Heading 1/2/3 styles;
2. adds ``w:pStyle`` to every paragraph that already carries an ``outlineLvl``
   so Word's navigation pane, TOC field and heading styles all line up;
3. replaces the empty ``<w:sectPr/>`` with A4 page setup using the margins
   suggested in the 876H1Z handbook (top/bottom 25 mm, left 40 mm for binding,
   right 15 mm);
4. replaces any paragraph containing the marker ``[[TOC]]`` with a live Word
   table-of-contents field (``--toc``, on by default);
5. adds a page-number footer in the format suggested by the handbook and
   suppresses it on the cover page (``--footer``, on by default);
6. turns on ``w:updateFields`` so Word refreshes the TOC and page fields when
   the document is first opened (``--update-fields``, on by default).

Usage:
    python docx_format.py <file.docx> [<file.docx> ...]
    python docx_format.py <file.docx> --no-footer --no-toc
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

STYLES_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults>
<w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun" w:cs="Times New Roman"/>
<w:sz w:val="21"/><w:szCs w:val="21"/>
<w:lang w:val="en-GB" w:eastAsia="zh-CN"/>
</w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:jc w:val="both"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
<w:name w:val="Normal"/><w:qFormat/>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading1">
<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:outlineLvl w:val="0"/><w:spacing w:before="360" w:after="200"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading2">
<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:outlineLvl w:val="1"/><w:spacing w:before="280" w:after="140"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:styleId="Heading3">
<w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:outlineLvl w:val="2"/><w:spacing w:before="240" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
</w:style>
</w:styles>"""

SECT_PR = (
    '<w:sectPr>'
    '{footer_ref}'
    '<w:pgSz w:w="11906" w:h="16838"/>'
    '<w:pgMar w:top="1417" w:right="850" w:bottom="1417" w:left="2268" '
    'w:header="851" w:footer="720" w:gutter="0"/>'
    '<w:cols w:space="708"/>'
    '{title_pg}'
    '</w:sectPr>'
)

TOC_PARAGRAPH = (
    '<w:p>'
    '<w:pPr><w:keepNext/><w:spacing w:after="160"/><w:rPr><w:sz w:val="21"/></w:rPr></w:pPr>'
    '<w:fldSimple w:instr=" TOC \\o &quot;1-3&quot; \\h \\z \\u ">'
    '<w:r><w:rPr><w:i/><w:sz w:val="21"/></w:rPr>'
    '<w:t>Table of contents \u2014 open in Word and update the field (Ctrl+A then F9) '
    'to generate entries and page numbers.</w:t></w:r>'
    '</w:fldSimple>'
    '</w:p>'
)

FOOTER_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W}">
<w:p>
<w:pPr><w:jc w:val="center"/><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>
<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">876H1Z Interim report    Page </w:t></w:r>
<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>
<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve"> of </w:t></w:r>
<w:fldSimple w:instr=" NUMPAGES "><w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>
<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">    Candidate number: ________________</w:t></w:r>
</w:p>
</w:ftr>"""

OUTLINE_TO_STYLE = {"0": "Heading1", "1": "Heading2", "2": "Heading3"}


def _apply_pstyle(match: re.Match[str]) -> str:
    para = match.group(0)
    level = re.search(r'<w:outlineLvl w:val="(\d)"', para)
    if not level:
        return para
    style = OUTLINE_TO_STYLE.get(level.group(1))
    if not style or "<w:pStyle" in para:
        return para
    return re.sub(r"<w:pPr>", f'<w:pPr><w:pStyle w:val="{style}"/>', para, count=1)


def _replace_toc_marker(doc: str) -> tuple[str, int]:
    pattern = re.compile(r"<w:p\b[^>]*>(?:(?!</w:p>).)*\[\[TOC\]\](?:(?!</w:p>).)*</w:p>", re.S)
    found = len(pattern.findall(doc))
    if not found:
        return doc, 0
    # NB: pass a callable so re does not interpret backslash escapes in the
    # replacement (the TOC field instruction contains "\o", "\h", "\z", "\u").
    return pattern.sub(lambda _m: TOC_PARAGRAPH, doc, count=1), 1


def process(path: Path, toc: bool = True, footer: bool = True, update_fields: bool = True) -> dict:
    with zipfile.ZipFile(path) as zin:
        entries = [(i, zin.read(i.filename)) for i in zin.infolist()]

    stats = {"paragraphs": 0, "headings": 0, "sectPr": 0, "toc": 0, "footer": 0, "updateFields": 0}
    names = {i.filename for i, _ in entries}

    # pick a footer relationship id that is not already used
    rels_name = "word/_rels/document.xml.rels"
    rels = next((d.decode("utf-8") for i, d in entries if i.filename == rels_name), "")
    used = set(re.findall(r'Id="([^"]+)"', rels))
    footer_id = "rIdFtr1"
    while footer_id in used:
        footer_id = footer_id + "x"

    out: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in entries:
        name = info.filename

        if name == "word/document.xml":
            doc = data.decode("utf-8")
            doc, n = re.subn(r"<w:p[ >].*?</w:p>", _apply_pstyle, doc, flags=re.S)
            stats["paragraphs"] = n
            stats["headings"] = doc.count("<w:pStyle")

            if toc:
                doc, stats["toc"] = _replace_toc_marker(doc)

            use_footer = footer and "<w:sectPr/>" in doc
            if "<w:sectPr/>" in doc:
                doc = doc.replace(
                    "<w:sectPr/>",
                    SECT_PR.format(
                        footer_ref=(
                            f'<w:footerReference w:type="default" r:id="{footer_id}"/>' if use_footer else ""
                        ),
                        title_pg="<w:titlePg/>" if use_footer else "",
                    ),
                )
                stats["sectPr"] = 1
                stats["footer"] = 1 if use_footer else 0
            data = doc.encode("utf-8")

        elif name == "word/styles.xml":
            data = STYLES_XML.encode("utf-8")

        elif name == rels_name and footer:
            text = data.decode("utf-8")
            if "footer1.xml" not in text:
                rel = (
                    f'<Relationship Id="{footer_id}" Type="{R}/footer" Target="footer1.xml" />'
                )
                text = text.replace("</Relationships>", rel + "</Relationships>")
            data = text.encode("utf-8")

        elif name == "[Content_Types].xml" and footer:
            text = data.decode("utf-8")
            if "footer1.xml" not in text:
                override = (
                    '<Override PartName="/word/footer1.xml" ContentType="application/'
                    'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                )
                text = text.replace("</Types>", override + "</Types>")
            data = text.encode("utf-8")

        elif name == "word/settings.xml" and update_fields:
            text = data.decode("utf-8")
            if "updateFields" not in text:
                text = text.replace(
                    "</w:settings>", '<w:updateFields w:val="true"/></w:settings>'
                )
                stats["updateFields"] = 1
            data = text.encode("utf-8")

        out.append((info, data))

    if footer and "word/footer1.xml" not in names:
        fname = "word/footer1.xml"
        info = zipfile.ZipInfo(fname)
        info.compress_type = zipfile.ZIP_DEFLATED
        out.append((info, FOOTER_XML.encode("utf-8")))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in out:
            zout.writestr(info, data)
    shutil.move(str(tmp), str(path))
    return stats


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        raise SystemExit(__doc__)
    for arg in args:
        p = Path(arg).resolve()
        result = process(
            p,
            toc="--no-toc" not in flags,
            footer="--no-footer" not in flags,
            update_fields="--no-update-fields" not in flags,
        )
        print(f"{p.name}: {result}")
