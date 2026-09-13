# -*- coding: utf-8 -*-
"""
make_docx.py -- build an editable DOCX version of the companion paper.

Pipeline
  1. Parse paper.aux for the authoritative label -> printed-number map.
  2. Pre-process the LaTeX sources into one self-contained .tex that pandoc
     can read natively (citations expanded to author-year brackets, \\ref/
     \\eqref replaced by printed numbers, listings inlined with bold caption
     paragraphs, subfigures flattened, equations numbered via \\qquad\\text).
  3. Build a reference.docx (A4, Times/SimSun, black headings, bordered
     tables, page-number footer) from pandoc's default.
  4. Run pandoc (native OMML equations).
  5. Post-process document.xml (native TOC field, page breaks).

Usage:  python make_docx.py
Output: paper/paper_editable.docx
"""
import io, os, re, sys, subprocess, zipfile, shutil, tempfile

sys.stdout.reconfigure(encoding="utf-8")
PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/
ROOT = os.path.dirname(PAPER)                                        # repo root
PANDOC = os.path.join(ROOT, "work", "tools", "pandoc_x", "pandoc-3.11", "pandoc.exe")
OUT_DOCX = os.path.join(PAPER, "paper_editable.docx")
WORKDIR = os.path.join(ROOT, "work", "docx_build")

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def strip_comments(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(s[i:i+2]); i += 2; continue
        if c == "%":
            j = s.find("\n", i)
            if j < 0: break
            i = j; continue
        out.append(c); i += 1
    return "".join(out)

def read(p):
    return io.open(p, encoding="utf-8").read()

def find_closing(s, start):
    """s[start] == '{' -> index of matching '}'"""
    depth, i = 0, start
    while i < len(s):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0: return i
        i += 1
    raise ValueError("unbalanced braces")

def grab_group(s, kw):
    """content of \\kw{...} starting at s.index('\\kw')"""
    i = s.index("\\" + kw)
    j = s.index("{", i)
    k = find_closing(s, j)
    return s[j+1:k], i, k

# --------------------------------------------------------------------------
# 1. label map from paper.aux
# --------------------------------------------------------------------------
def load_aux():
    aux = {}
    src = read(os.path.join(PAPER, "paper.aux"))
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}", src):
        aux[m.group(1)] = {"num": m.group(2), "page": m.group(3)}
    # type info lives in the 4th group when present
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\}", src):
        aux[m.group(1)]["type"] = m.group(5)
    return aux

AUX = load_aux()

def printed(label):
    e = AUX.get(label)
    if not e:
        print("  !! unknown label:", label)
        return "??"
    t = e.get("type", "")
    fam = t.split(".")[0]
    rest = t.split(".", 1)[1] if "." in t else e["num"]
    return rest

# --------------------------------------------------------------------------
# 2. citation map from parts/10_bib.tex
# --------------------------------------------------------------------------
def load_bib():
    src = read(os.path.join(PAPER, "parts", "10_bib.tex"))
    keys, order = {}, []
    for m in re.finditer(r"\\bibitem\[([^\]]*)\]\{([^}]+)\}", src):
        keys[m.group(2)] = m.group(1)
        order.append((m.group(1), m.group(2)))
    return keys, order

CITE, BIBORDER = load_bib()

# --------------------------------------------------------------------------
# 3. image resolution (append .png / map webp -> png)
# --------------------------------------------------------------------------
IMGDIRS = [os.path.join(PAPER, "figures", "png"),
           os.path.join(PAPER, "figures"),
           PAPER]
_img_cache = {}

def resolve_image(name):
    if name in _img_cache:
        return _img_cache[name]
    base, ext = os.path.splitext(name)
    cand = None
    if ext:
        for d in IMGDIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p): cand = p; break
    else:
        for e in (".png", ".pdf", ".webp", ".jpg"):
            for d in IMGDIRS:
                p = os.path.join(d, name + e)
                if os.path.isfile(p): cand = p; break
            if cand: break
    if cand is None:
        print("  !! image not found:", name)
        _img_cache[name] = name
        return name
    if cand.lower().endswith(".webp"):           # webp -> png for Word
        from PIL import Image
        outp = os.path.join(WORKDIR, "imgs", base.replace("/", "_") + ".png")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        if not os.path.isfile(outp):
            Image.open(cand).convert("RGB").save(outp)
        cand = outp
    elif cand.lower().endswith(".pdf"):           # vector pdf -> png for Word
        from PIL import Image
        outp = os.path.join(WORKDIR, "imgs", base.replace("/", "_") + ".png")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        if not os.path.isfile(outp):
            subprocess.run([r"C:\Users\AICoding\.cache\codex-runtimes"
                            r"\codex-primary-runtime\dependencies\native\poppler"
                            r"\Library\bin\pdftoppm.exe",
                            "-r", "200", "-singlefile", "-png", cand,
                            outp[:-4]], check=True)
        cand = outp
    _img_cache[name] = cand
    return cand

# --------------------------------------------------------------------------
# 4. transforms
# --------------------------------------------------------------------------
DISPLAY_SEQ = []   # per display equation: printed number or None

def sub_refs(s):
    s = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(%s)" % printed(m.group(1)), s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: printed(m.group(1)), s)
    s = re.sub(r"\\cite\{([^}]+)\}",
               lambda m: "[" + "; ".join(CITE.get(k.strip(), k.strip())
                                         for k in m.group(1).split(",")) + "]", s)
    return s

def sub_images(s):
    def rep(m):
        opts, name = m.group(1) or "", m.group(2)
        p = resolve_image(name)
        return "\\includegraphics%s{%s}" % (opts, p.replace("\\", "/"))
    return re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", rep, s)

def txteq(num):
    return r"\qquad\text{(%s)}" % num

def sub_equations(s):
    """numbered equation/align -> \\[ ... \\] with \\qquad\\text numbers"""
    def eq_rep(m):
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        body = re.sub(r"\\label\{[^}]+\}", "", body)
        num = printed(lab.group(1)) if lab else None
        DISPLAY_SEQ.append(num)
        return "\\[\n%s%s\n\\]" % (body.strip(), txteq(num) if num else "")
    s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq_rep, s, flags=re.S)

    def al_rep(m):
        body = m.group(1)
        lines = body.split("\\\\")
        out = []
        for ln in lines:
            lab = re.search(r"\\label\{([^}]+)\}", ln)
            ln2 = re.sub(r"\\label\{[^}]+\}", "", ln)
            num = printed(lab.group(1)) if lab else None
            DISPLAY_SEQ.append(num)
            out.append(ln2.rstrip() + (txteq(num) if num else " \\notag"))
        return "\\begin{align*}\n" + " \\\\\n".join(out) + "\n\\end{align*}"
    s = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", al_rep, s, flags=re.S)
    return s

def sub_code(s, counter):
    """lstlisting with caption opts + listing floats with lstinputlisting"""
    def lst_rep(m):
        opts, body = m.group(1) or "", m.group(2)
        cap = ""
        cm = re.search(r"caption=\{([^}]*)\}", opts)
        if cm: cap = cm.group(1)
        lm = re.search(r"label=\{([^}]+)\}", opts)
        n = printed(lm.group(1)) if lm and lm.group(1) in AUX else str(counter[0] + 1)
        counter[0] = int(n)
        head = "\n\n\\noindent\\textbf{代码 %s：%s}\n\n" % (n, cap) if cap else "\n\n"
        return head + "\\begin{lstlisting}\n" + body + "\\end{lstlisting}\n"
    s = re.sub(r"\\begin\{lstlisting\}(\[[^\]]*\])?(.*?)\\end\{lstlisting\}",
               lst_rep, s, flags=re.S)

    def flt_rep(m):
        body = m.group(1)
        capm = re.search(r"\\caption\{", body)
        cap = ""
        if capm:
            j = s.index  # noqa
            i0 = body.index("\\caption{")
            k = find_closing(body, i0 + len("\\caption"))
            cap = body[i0 + len("\\caption{"):k]
        labm = re.search(r"\\label\{([^}]+)\}", body)
        n = printed(labm.group(1)) if labm and labm.group(1) in AUX else str(counter[0] + 1)
        counter[0] = int(n)
        inm = re.search(r"\\lstinputlisting\[([^\]]*)\]\{([^}]+)\}|\\lstinputlisting\{([^}]+)\}", body)
        if not inm:
            return ""
        opts = inm.group(1) or ""
        path = (inm.group(2) or inm.group(3)).strip()
        first = re.search(r"firstline=(\d+)", opts)
        last = re.search(r"lastline=(\d+)", opts)
        lines = read(os.path.join(PAPER, path.replace("/", os.sep))).split("\n")
        a = int(first.group(1)) if first else 1
        b = int(last.group(1)) if last else len(lines)
        code = "\n".join(lines[a-1:b])
        return ("\n\n\\noindent\\textbf{代码 %s：%s}\n\n"
                "\\begin{lstlisting}\n%s\n\\end{lstlisting}\n") % (n, cap, code)
    s = re.sub(r"\\begin\{listing\}\[[^\]]*\](.*?)\\end\{listing\}", flt_rep, s, flags=re.S)
    return s

def sub_figures(s):
    """number captions; flatten subfigure groups into images + bold captions"""
    def fig_rep(m):
        body = m.group(1)
        labm = re.search(r"\\label\{(fig:[^}]+)\}", body)
        if "\\begin{subfigure}" in body:
            return flatten_subfigs(body, labm.group(1) if labm else None)
        num = printed(labm.group(1)) if labm else None
        capm = re.search(r"\\caption\{", body)
        if capm and num:
            i0 = body.index("\\caption{")
            k = find_closing(body, i0 + len("\\caption"))
            body = body[:i0] + "\\caption{图 %s：%s}" % (num, body[i0+len("\\caption{"):k]) + body[k+1:]
        body = re.sub(r"\\centering", "", body)
        return "\\begin{figure}[H]\n" + body + "\n\\end{figure}"
    s = re.sub(r"\\begin\{figure\}\[[^\]]*\](.*?)\\end\{figure\}", fig_rep, s, flags=re.S)
    return s

def flatten_subfigs(body, figlabel):
    num = printed(figlabel) if figlabel else "??"
    capm = re.search(r"\\caption\{", body)
    maincap = ""
    if capm:
        i0 = body.index("\\caption{")
        k = find_closing(body, i0 + len("\\caption"))
        maincap = body[i0+len("\\caption{"):k]
    parts = []
    letter = ord("a")
    for sm in re.finditer(r"\\begin\{subfigure\}[^\{]*\{(.*?)\}(.*?)\\end\{subfigure\}", body, flags=re.S):
        sub = sm.group(2)
        im = re.search(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", sub)
        cm = re.search(r"\\caption\{", sub)
        subcap = ""
        if cm:
            j0 = sub.index("\\caption{")
            jk = find_closing(sub, j0 + len("\\caption"))
            subcap = sub[j0+len("\\caption{"):jk]
        w = "[width=0.86\\linewidth]" if sm.group(1) != "\\linewidth" else "[width=0.95\\linewidth]"
        parts.append("\\begin{center}\n\\includegraphics%s{%s}\n\\end{center}\n\n"
                     "\\noindent\\textbf{(%s) %s}\n\n" %
                     (w, resolve_image(im.group(2)).replace("\\", "/") if im else "",
                      chr(letter), subcap))
        letter += 1
    out = "\n".join(parts)
    out += "\\noindent\\textbf{图 %s：%s}\n\n" % (num, maincap)
    return out

def sub_tables(s):
    def rep(m):
        body = m.group(1)
        labm = re.search(r"\\label\{(tab:[^}]+)\}", body)
        num = printed(labm.group(1)) if labm else None
        capm = re.search(r"\\caption\{", body)
        if capm and num:
            i0 = body.index("\\caption{")
            k = find_closing(body, i0 + len("\\caption"))
            body = body[:i0] + "\\caption{表 %s：%s}" % (num, body[i0+len("\\caption{"):k]) + body[k+1:]
        body = re.sub(r"\\centering|\\small|\\footnotesize|\\scriptsize", "", body)
        return "\\begin{table}[H]\n" + body + "\n\\end{table}"
    s = re.sub(r"\\begin\{table\}\[[^\]]*\](.*?)\\end\{table\}", rep, s, flags=re.S)
    # appendix longtables (auxiliary, uncaptioned)
    s = s.replace("\\auxkeepcounter{900}\n", "").replace("\\auxkeepcounter{920}\n", "")
    s = re.sub(r"\\auxkeepcounter\{\d+\}|\\auxrestorecounter|\\setcounter\{[^}]*\}\{[^}]*\}", "", s)
    s = s.replace("\\begin{longtable}", "\\begin{tabular}").replace("\\end{longtable}", "\\end{tabular}")
    return s

def sub_bib(s):
    if "\\begin{thebibliography}" not in s:
        return s
    i0 = s.index("\\begin{thebibliography}")
    i1 = s.index("\\end{thebibliography}") + len("\\end{thebibliography}")
    head = s[:i0] + "\\subsection*{参考文献}\n"
    body = s[i0:i1]
    items = re.split(r"\\bibitem", body)[1:]
    out = []
    for it in items:
        lm = re.match(r"\[([^\]]*)\]\{[^}]+\}", it)
        label = lm.group(1) if lm else ""
        text = it[lm.end():] if lm else it
        if "\\end{thebibliography}" in text:
            break
        text = text.replace("\\small", "").strip()
        out.append("\\noindent\\textbf{[%s]}\\quad %s\\par\\medskip\n" % (label, text))
    return head + "\n".join(out) + s[i1:]

def process_part(name, counter):
    s = read(os.path.join(PAPER, name + ".tex"))
    s = strip_comments(s)
    # inline tables
    s = re.sub(r"\\input\{(tables/[^}]+)\}",
               lambda m: read(os.path.join(PAPER, m.group(1) + ".tex")), s)
    s = re.sub(r"\{\\rm\s*([^{}]*)\}", r"{\\mathrm{\1}}", s)
    s = re.sub(r"\\rm\s+([A-Za-z]+)", r"\\mathrm{\1}", s)
    s = sub_code(s, counter)
    s = sub_equations(s)
    s = sub_figures(s)
    s = sub_tables(s)
    s = sub_bib(s)
    s = sub_images(s)
    s = sub_refs(s)
    # cleanup
    s = re.sub(r"^\s*\\\\\[5pt\]\s*$", "", s, flags=re.M)
    s = s.replace("\\hfill", " ")
    return s

# --------------------------------------------------------------------------
# 5. preamble of the pandoc input
# --------------------------------------------------------------------------
def build_input_tex():
    os.makedirs(WORKDIR, exist_ok=True)
    src = read(os.path.join(PAPER, "paper.tex"))
    # collect math shorthand + theorem definitions from the marked block
    blk = re.search(r"% -- math shorthands.*?(?=\n\\theoremstyle)", src, flags=re.S).group(0)
    macros = [l for l in blk.split("\n") if l.startswith("\\newcommand")]
    macros += ["\\theoremstyle{plain}",
               "\\newtheorem{proposition}{命题}",
               "\\newtheorem{lemma}{引理}",
               "\\theoremstyle{definition}",
               "\\newtheorem{remark}{注记}"]
    title = re.search(r"\\title\{(.*?)\}\s*\n\\author", src, flags=re.S).group(1)
    author = re.search(r"\\author\{(.*?)\}\s*\n\\date", src, flags=re.S).group(1)
    date = re.search(r"\\date\{(.*?)\}\s*\\begin\{document\}", src, flags=re.S).group(1)
    abst = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, flags=re.S).group(1)

    order = ["parts/01_intro", "parts/02_theory", "parts/03_numerics",
             "parts/04_impl", "parts/05_verify", "parts/06_results",
             "parts/07_discussion", "parts/08_conclusion", "parts/10_bib",
             "parts/09_appendix"]
    counter = [0]
    body = "\n\n".join(process_part(p, counter) for p in order)

    tex = []
    tex.append("\\documentclass{article}")
    tex.append("\\usepackage{amsmath,amssymb}")
    tex.append("\\usepackage{graphicx}")
    tex.append("\\usepackage{float}")
    tex.append("\n".join(macros))
    tex.append("\\title{%s}" % title)
    tex.append("\\author{%s}" % author)
    tex.append("\\date{%s}" % date)
    tex.append("\\begin{document}")
    tex.append("\\maketitle")
    tex.append("\\begin{abstract}%s\\end{abstract}" % abst)
    tex.append(body)
    tex.append("\\end{document}")
    out = os.path.join(WORKDIR, "paper_docx.tex")
    io.open(out, "w", encoding="utf-8").write("\n".join(tex))
    return out

# --------------------------------------------------------------------------
# 6. reference.docx
# --------------------------------------------------------------------------
FOOTER_XML = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
 '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
 '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
 '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
 '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
 '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
 '</w:p></w:ftr>')

def style_block(styles, style_id):
    m = re.search(r'<w:style [^>]*w:styleId="%s">.*?</w:style>' % style_id, styles, flags=re.S)
    return m

def build_reference():
    os.makedirs(WORKDIR, exist_ok=True)
    base = os.path.join(WORKDIR, "ref_base.docx")
    if not os.path.isfile(base):
        data = subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                              check=True, capture_output=True).stdout
        io.open(base, "wb").write(data)
    out = os.path.join(WORKDIR, "reference_custom.docx")
    zin = zipfile.ZipFile(base)
    names = zin.namelist()
    files = {n: zin.read(n) for n in names}
    zin.close()

    styles = files["word/styles.xml"].decode("utf-8")

    # document defaults: Times/SimSun 11pt
    styles = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
        '<w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
        ' w:eastAsia="SimSun" w:cs="Times New Roman"/>'
        '<w:sz w:val="22"/><w:szCs w:val="22"/>'
        '<w:lang w:val="en-US" w:eastAsia="zh-CN" w:bidi="ar-SA"/>'
        '</w:rPr></w:rPrDefault>', styles, flags=re.S)

    # headings: black + SimHei
    def fix_heading(m):
        blk = m.group(0)
        blk = re.sub(r'w:asciiTheme="majorHAnsi"', 'w:ascii="Times New Roman"', blk)
        blk = re.sub(r'w:hAnsiTheme="majorHAnsi"', 'w:hAnsi="Times New Roman"', blk)
        blk = re.sub(r'w:eastAsiaTheme="majorEastAsia"', 'w:eastAsia="SimHei"', blk)
        blk = re.sub(r'w:cstheme="majorBidi"', 'w:cs="Times New Roman"', blk)
        blk = re.sub(r'<w:color w:val="[0-9A-Fa-f]{6}"/>', '<w:color w:val="000000"/>', blk)
        return blk
    styles = re.sub(r'<w:style [^>]*w:styleId="Heading\d">.*?</w:style>', fix_heading, styles, flags=re.S)

    add = ""
    if not style_block(styles, "SourceCode"):
        add += ('<w:style w:type="paragraph" w:styleId="SourceCode">'
                '<w:name w:val="Source Code"/><w:basedOn w:val="Normal"/>'
                '<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F7F7F9"/>'
                '<w:spacing w:before="120" w:after="120"/><w:contextualSpacing/>'
                '<w:spacing w:line="240" w:lineRule="auto"/></w:pPr>'
                '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="SimSun"/>'
                '<w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style>')
    if not style_block(styles, "VerbatimChar"):
        add += ('<w:style w:type="character" w:styleId="VerbatimChar">'
                '<w:name w:val="Verbatim Char"/>'
                '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                '<w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>')
    for sid, nm in (("ImageCaption", "Image Caption"), ("TableCaption", "Table Caption")):
        if not style_block(styles, sid):
            add += ('<w:style w:type="paragraph" w:styleId="%s">'
                    '<w:name w:val="%s"/><w:basedOn w:val="Normal"/>'
                    '<w:pPr><w:spacing w:before="60" w:after="120"/>'
                    '<w:jc w:val="center"/></w:pPr>'
                    '<w:rPr><w:b/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>') % (sid, nm)
    # bordered table style
    if style_block(styles, "Table"):
        styles = re.sub(r'(<w:style [^>]*w:styleId="Table"><w:name w:val="Table"/>)',
            r'\1<w:tblPr><w:tblBorders>'
            '<w:top w:val="single" w:sz="8" w:color="444444"/>'
            '<w:bottom w:val="single" w:sz="8" w:color="444444"/>'
            '<w:insideH w:val="single" w:sz="4" w:color="BBBBBB"/>'
            '</w:tblBorders></w:tblPr>', styles, count=1)
    if add:
        styles = styles.replace("</w:styles>", add + "</w:styles>")
    files["word/styles.xml"] = styles.encode("utf-8")

    # A4 sectPr + footer
    doc = files["word/document.xml"].decode("utf-8")
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFtr1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1361" w:right="1304" w:bottom="1361" w:left="1304"'
            ' w:header="851" w:footer="709" w:gutter="0"/>'
            '<w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>')
    if "<w:sectPr" in doc:
        doc = re.sub(r"<w:sectPr.*?</w:sectPr>", sect, doc, flags=re.S)
    else:
        doc = doc.replace("</w:body>", sect + "</w:body>")
    files["word/document.xml"] = doc.encode("utf-8")

    files["word/footer1.xml"] = FOOTER_XML.encode("utf-8")
    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    rels = rels.replace("</Relationships>",
        '<Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')
    files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    ct = files["[Content_Types].xml"].decode("utf-8")
    ct = ct.replace("</Types>",
        '<Override PartName="/word/footer1.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    files["[Content_Types].xml"] = ct.encode("utf-8")

    zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    for n, d in files.items():
        zout.writestr(n, d)
    zout.close()
    return out

# --------------------------------------------------------------------------
# 7. post-process output document.xml
# --------------------------------------------------------------------------
TOC_BLOCK = (
 '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
 '<w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>目　录</w:t></w:r></w:p>'
 '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
 '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
 '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
 '<w:r><w:t>（在 Word 中：引用 → 更新目录，或右键此处选择“更新域”）</w:t></w:r>'
 '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
 '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

def postprocess(docx_path):
    tmp = docx_path + ".tmp"
    zin = zipfile.ZipFile(docx_path)
    files = {n: zin.read(n) for n in zin.namelist()}
    zin.close()
    doc = files["word/document.xml"].decode("utf-8")
    i = doc.find('w:val="Heading1"')
    if i >= 0:
        j = doc.rfind("<w:p ", 0, i)
        if j < 0: j = doc.rfind("<w:p>", 0, i)
        doc = doc[:j] + TOC_BLOCK + doc[j:]
    files["word/document.xml"] = doc.encode("utf-8")
    zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for n, d in files.items():
        zout.writestr(n, d)
    zout.close()
    shutil.move(tmp, docx_path)

# --------------------------------------------------------------------------
def main():
    if os.path.isdir(WORKDIR):
        shutil.rmtree(WORKDIR, ignore_errors=True)
    tex = build_input_tex()
    print("preprocessed tex:", tex, "(%d numbered displays)" % len(DISPLAY_SEQ))
    ref = build_reference()
    print("reference docx:", ref)
    cmd = [PANDOC, tex, "-f", "latex", "-t", "docx", "-o", OUT_DOCX,
           "--reference-doc", ref,
           "--resource-path", PAPER,
           "--number-sections",
           "--metadata", "lang=zh-CN"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("pandoc exit:", r.returncode)
    if r.stderr:
        print(r.stderr[-4000:])
    if r.returncode != 0:
        sys.exit(1)
    postprocess(OUT_DOCX)
    print("OK ->", OUT_DOCX, os.path.getsize(OUT_DOCX), "bytes")

if __name__ == "__main__":
    main()
