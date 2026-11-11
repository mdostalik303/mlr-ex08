"""Generate the QR code that index.html shows at the bottom.

Everything happens on this machine: `qrcode` builds the matrix, we write the SVG
ourselves and inline it into index.html between the QR:BEGIN / QR:END markers.
No QR web service is involved, at build time or at run time.

    python -I make_qr.py                      # write the SVG into index.html
    python -I make_qr.py --check <url>        # decode the shipped SVG and compare

Uses a venv outside the served folder ( Pillow, qrcode, zxing-cpp ).
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import qrcode
import qrcode.constants
import zxingcpp
from PIL import Image

HERE = Path(__file__).resolve().parent
HTML = HERE / "index.html"
MARK = re.compile(r"<!--QR:BEGIN-->.*?<!--QR:END-->", re.S)
PLACEHOLDER = "<!--QR_SVG-->"


def page_url_from_html() -> str:
    m = re.search(r'const PAGE_URL = "([^"]+)"', HTML.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("no PAGE_URL constant in index.html")
    return m.group(1)


def build_svg(data: str) -> str:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    rows = qr.get_matrix()
    n = len(rows)
    # One horizontal run per continuous stretch of dark modules keeps the path short.
    parts = []
    for y, row in enumerate(rows):
        x = 0
        while x < n:
            if row[x]:
                x0 = x
                while x < n and row[x]:
                    x += 1
                parts.append(f"M{x0} {y}h{x - x0}v1h-{x - x0}z")
            x += 1
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="168" height="168" '
        f'role="img" aria-label="QR code for {data}" shape-rendering="crispEdges">'
        f'<rect width="{n}" height="{n}" fill="#ffffff"/>'
        f'<path fill="#000000" d="{" ".join(parts)}"/></svg>'
    )
    return svg


def decode(svg: str) -> str:
    """Render the same matrix to a bitmap and read it back with a second, independent library."""
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    n = int(m.group(1))
    # Rebuild the bitmap from the path's runs, exactly as a browser would paint it.
    scale = 8
    img = np.full((n * scale, n * scale), 255, dtype=np.uint8)
    for run in re.findall(r"M(\d+) (\d+)h(\d+)v1", svg):
        x0, y, w = int(run[0]), int(run[1]), int(run[2])
        img[y * scale:(y + 1) * scale, x0 * scale:(x0 + w) * scale] = 0
    results = zxingcpp.read_barcodes(Image.fromarray(img))
    if not results:
        raise SystemExit("the generated QR could not be decoded at all")
    return results[0].text


def current_svg() -> str | None:
    m = MARK.search(HTML.read_text(encoding="utf-8"))
    return m.group(0) if m else None


if __name__ == "__main__":
    url = page_url_from_html()

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        want = sys.argv[2] if len(sys.argv) > 2 else url
        inside = current_svg()
        if not inside:
            raise SystemExit("index.html has no QR yet")
        got = decode(inside)
        print("QR encodes :", got)
        print("live URL   :", want)
        raise SystemExit(0 if got == want else "MISMATCH")

    svg = build_svg(url)
    encoded = decode(svg)
    assert encoded == url, f"QR round-trip gave {encoded!r}"
    html = HTML.read_text(encoding="utf-8")
    block = f"<!--QR:BEGIN-->{svg}<!--QR:END-->"
    if PLACEHOLDER in html:
        html = html.replace(PLACEHOLDER, block)
    elif MARK.search(html):
        html = MARK.sub(lambda _: block, html, count=1)
    else:
        raise SystemExit("neither the placeholder nor a QR block is in index.html")
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote a {len(svg)}-byte QR SVG for {url} (decoded back as {encoded!r})")
