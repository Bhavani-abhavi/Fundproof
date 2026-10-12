"""Take screenshots of the static frontend build (frontend/out) with Playwright + Chromium.

Run after `npm run build` (from the repository root):

    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
        uv run --with playwright==1.56.0 python frontend/scripts/screenshots.py

What it does:
1. Serves frontend/out with Python's built-in `http.server` on a free local port (no backend), so /api/* answers
   with an HTML 404 page. This is the "static preview" case: pages must show a friendly offline state.
2. Opens each page at 1280 px and 375 px wide, in light and dark color schemes, and saves full-page PNGs to
   docs/web/screenshots/.
3. With `--live-url http://127.0.0.1:8765` it also drives a RUNNING local server started with dev login on
   (`UWC_DEV_AUTH=1 uv run uvicorn uwcopilot.web.app:app --port 8765`): it signs in through the DEV LOGIN form as
   a SYNTHETIC demo founder and a SYNTHETIC demo senior lender, saves `*-devserver-*` screenshots, signs out, and
   fails if the browser logged any console error (for example a Content-Security-Policy violation).

No internet access is needed; the live mode only talks to the local server you pass.
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from playwright.sync_api import Browser, ConsoleMessage, Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "frontend" / "out"
DEFAULT_SCREENSHOT_DIR = REPO_ROOT / "docs" / "web" / "screenshots"

WIDTHS = (1280, 375)
COLOR_SCHEMES = ("light", "dark")

# (file name stem, path). Trailing slashes match the static export (next.config.ts: trailingSlash: true).
PAGES = (
    ("landing", "/"),
    ("signin", "/signin/"),
    ("studio", "/studio/"),
    ("lender", "/lender/"),
    ("about", "/about/"),
)


def free_port() -> int:
    """Ask the OS for an unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def static_server(directory: Path) -> Iterator[str]:
    """Serve `directory` with `python -m http.server` and yield its base URL."""
    port = free_port()
    command = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(directory)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen(base_url + "/", timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        yield base_url
    finally:
        process.terminate()
        process.wait(timeout=10)


class ConsoleWatcher:
    """Collects browser console errors and uncaught page errors (for example CSP violations)."""

    def __init__(self, page: Page) -> None:
        self.errors: list[str] = []
        page.on("console", self._on_console)
        page.on("pageerror", lambda error: self.errors.append(f"pageerror: {error}"))

    def _on_console(self, message: ConsoleMessage) -> None:
        if message.type == "error":
            self.errors.append(f"console: {message.text}")


def dev_login(page: Page, base_url: str, username: str, role_label: str) -> None:
    """Sign in through the DEV LOGIN form on /signin/ (the server must run with UWC_DEV_AUTH=1)."""
    page.goto(base_url + "/signin/")
    settle(page)
    page.get_by_label("Username").fill(username)
    page.get_by_role("combobox", name="Role").click()
    page.get_by_role("option", name=role_label).click()
    page.get_by_role("button", name="Sign in (dev)").click()
    page.wait_for_url("**/studio/" if role_label == "Founder" else "**/lender/")
    settle(page)


def sign_out(page: Page) -> None:
    """Click the header's Sign out button (desktop layout) and wait for the Sign in link."""
    page.get_by_role("button", name="Sign out").click()
    page.get_by_role("link", name="Sign in").first.wait_for()


def run_live(browser: Browser, base_url: str, screenshot_dir: Path) -> list[str]:
    """Drive a running dev server: dev login as a founder and as a senior lender, screenshot the signed-in
    pages, sign out. Returns the console errors seen (an empty list means none)."""
    errors: list[str] = []
    for width in WIDTHS:
        context = browser.new_context(viewport={"width": width, "height": 900}, color_scheme="light")
        page = context.new_page()
        watcher = ConsoleWatcher(page)
        page.goto(base_url + "/signin/")
        settle(page)
        page.screenshot(path=str(screenshot_dir / f"signin-devserver-{width}-light.png"), full_page=True)
        print(f"saved signin-devserver-{width}-light.png")

        dev_login(page, base_url, "demo-founder", "Founder")
        page.get_by_role("heading", name="Your plans").wait_for()
        page.screenshot(path=str(screenshot_dir / f"studio-devserver-founder-{width}-light.png"), full_page=True)
        print(f"saved studio-devserver-founder-{width}-light.png")

        if width >= 768:
            sign_out(page)
        else:
            context.clear_cookies()

        dev_login(page, base_url, "demo-senior", "Senior lender")
        page.get_by_text("Your queue is empty", exact=False).wait_for()
        page.screenshot(path=str(screenshot_dir / f"lender-devserver-senior-{width}-light.png"), full_page=True)
        print(f"saved lender-devserver-senior-{width}-light.png")
        errors.extend(watcher.errors)
        context.close()
    return errors


def settle(page: Page) -> None:
    """Wait until fonts are loaded and the session request has finished (no skeletons left)."""
    page.wait_for_load_state("networkidle")
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(300)


def capture(
    page: Page, base_url: str, path: str, file_path: Path, *, open_menu: bool = False, open_drawer: bool = False
) -> None:
    """Load one page and save a full-page screenshot (optionally with the mobile menu or the drawer open)."""
    page.goto(base_url + path)
    settle(page)
    full_page = True
    if open_menu:
        page.get_by_role("button", name="Open menu").click()
        page.wait_for_timeout(300)
        full_page = False
    if open_drawer:
        page.get_by_role("button", name="Show the source for Annual revenue").click()
        page.wait_for_timeout(300)
        full_page = False
    page.screenshot(path=str(file_path), full_page=full_page)
    print(f"saved {file_path.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="The static export to serve.")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--live-url", help="Base URL of a running dev server (UWC_DEV_AUTH=1) to sign in to.")
    args = parser.parse_args()

    if not (args.out_dir / "index.html").exists():
        print(f"{args.out_dir} has no index.html: run `npm run build` in frontend/ first.", file=sys.stderr)
        return 1
    args.screenshot_dir.mkdir(parents=True, exist_ok=True)

    with static_server(args.out_dir) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for scheme in COLOR_SCHEMES:
                for width in WIDTHS:
                    context = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
                    page = context.new_page()
                    for stem, path in PAGES:
                        capture(page, base_url, path, args.screenshot_dir / f"{stem}-{width}-{scheme}.png")
                    if width == 375:
                        target = args.screenshot_dir / f"menu-{width}-{scheme}.png"
                        capture(page, base_url, "/", target, open_menu=True)
                    else:
                        target = args.screenshot_dir / f"drawer-{width}-{scheme}.png"
                        capture(page, base_url, "/", target, open_drawer=True)
                    context.close()

            if args.live_url:
                errors = run_live(browser, args.live_url.rstrip("/"), args.screenshot_dir)
                for error in errors:
                    print(error, file=sys.stderr)
                if errors:
                    print(f"{len(errors)} browser console error(s) on the live server.", file=sys.stderr)
                    return 1
                print("live server: dev login, studio, lender and sign-out worked with no console errors")
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
