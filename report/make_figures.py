"""Figures for the capstone report that are not paper figures: dashboard screenshots, architecture diagrams, the
post-freeze dot plot and frames from the demo GIFs.

    python report/make_figures.py      # -> report/figures/*.png

Screenshots are taken from the offline dashboard (demo/index.html, light theme) with Playwright and Chromium. Set
CHROMIUM to the browser executable if Playwright's own browser is not installed.
"""
from __future__ import annotations

import asyncio
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
CHROMIUM = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SECTIONS = (("header", "dash_header"), ("#scorecard", "dash_scorecard"), ("#agents", "dash_agents"),
            ("#forgetting", "dash_forgetting"), ("#memory", "dash_memory"), ("#next", "dash_next"))


async def screenshots():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        kw = {"executable_path": CHROMIUM} if os.path.exists(CHROMIUM) else {}
        browser = await p.chromium.launch(**kw)
        page = await browser.new_page(viewport={"width": 1180, "height": 900}, color_scheme="light",
                                      device_scale_factor=2.5)
        await page.goto("file://" + os.path.join(ROOT, "demo", "index.html"))
        await page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")  # default sequence, light theme
        await page.reload()
        await page.wait_for_timeout(800)
        await (await page.query_selector("#archsvg svg")).screenshot(path=f"{OUT}/_arch_full.png")
        await (await page.query_selector("#memory .post")).screenshot(path=f"{OUT}/_postfreeze.png")
        for sel, name in SECTIONS:
            await (await page.query_selector(sel)).screenshot(path=f"{OUT}/{name}.png")
        await browser.close()


def crops():
    # the architecture SVG (viewBox y 20..392) holds the agent pipeline on top and the wake/sleep loop below it
    im = Image.open(f"{OUT}/_arch_full.png")
    w, h = im.size
    sy = h / 372
    top = im.crop((0, 0, w, int(220 * sy)))
    top.crop((0, 0, w, int(top.size[1] * 0.865))).save(f"{OUT}/arch_pipeline.png")
    low = im.crop((0, int(202 * sy), w, h))
    low.crop((0, int(low.size[1] * 0.125), w, low.size[1])).save(f"{OUT}/arch_wake_sleep.png")
    # the post-freeze follow-up's dot plot, without the surrounding text
    im = Image.open(f"{OUT}/_postfreeze.png")
    w, h = im.size
    im.crop((0, int(h * 0.435), w, int(h * 0.875))).save(f"{OUT}/c4_dots.png")
    for f in ("_arch_full.png", "_postfreeze.png"):
        os.remove(f"{OUT}/{f}")


def gif_frames():
    """3 x 3 grid per agent: the last frame of each recorded episode (episode ends are the frames held 1500 ms)."""
    font = ImageFont.truetype(FONT, 19)
    tasks = ["Task 1 (red ball)", "Task 2 (green key)", "Task 3 (blue box)"]
    for agent in ("naive", "sleep"):
        rows = []
        for k in range(3):
            g = Image.open(os.path.join(ROOT, "demo", "assets", f"fetch3_{agent}_task{k}.gif"))
            frames, durations = [], []
            for i in range(g.n_frames):
                g.seek(i)
                frames.append(g.convert("RGB"))
                durations.append(g.info.get("duration", 0))
            ends = [frames[i] for i, d in enumerate(durations) if d >= 1500]
            assert len(ends) == 3, (agent, k, len(ends))
            rows.append(ends)
        w, h = rows[0][0].size
        w2, h2, pad, lab = int(w * 0.6), int(h * 0.6), 16, 30
        sheet = Image.new("RGB", (3 * w2 + 4 * pad, 3 * (h2 + lab) + 4 * pad), "white")
        draw = ImageDraw.Draw(sheet)
        for r in range(3):
            for c in range(3):
                x, y = pad + c * (w2 + pad), pad + r * (h2 + lab + pad)
                draw.text((x, y), f"{tasks[r]}, episode {c + 1}", fill=(40, 40, 40), font=font)
                sheet.paste(rows[r][c].resize((w2, h2), Image.LANCZOS), (x, y + lab))
        sheet.save(f"{OUT}/gif_frames_{agent}.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    asyncio.run(screenshots())
    crops()
    gif_frames()
    print("figures:", sorted(os.listdir(OUT)))
