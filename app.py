"""
VitalScan Food Recognition — Gradio demo UI
"""
from __future__ import annotations

import os

import gradio as gr
import requests

API_URL = os.environ.get("VITALSCAN_API", "http://localhost:8000").rstrip("/")
TIMEOUT = 30

NUTRIENT_ROWS = [
    ("calories",  "Calories",  "kcal"),
    ("carbs_g",   "Carbs",     "g"),
    ("sugar_g",   "Sugar",     "g"),
    ("fat_g",     "Fat",       "g"),
    ("sodium_mg", "Sodium",    "mg"),
]

TIER_STYLE = {
    "high":   ("#1a7f37", "High confidence"),
    "medium": ("#b08800", "Medium confidence — confirm below"),
    "low":    ("#cf222e", "Low confidence — not recognized"),
}


def check_health() -> str:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        if data.get("model_ready"):
            return f"🟢 API online — model loaded ({API_URL})"
        return f"🟡 API online, but no model checkpoint loaded ({API_URL})"
    except requests.RequestException:
        return f"🔴 Cannot reach API at {API_URL} — is uvicorn running?"


def _loading_html() -> str:
    return """
<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
            padding:40px;gap:16px'>
  <div style='width:48px;height:48px;border:5px solid #e0e0e0;border-top-color:#1a7f37;
              border-radius:50%;animation:spin 0.9s linear infinite'></div>
  <p style='margin:0;color:#555;font-size:15px'>Analysing your food&hellip;</p>
</div>
<style>@keyframes spin{to{transform:rotate(360deg)}}</style>
"""


def scan_photo(image_path: str | None):
    if not image_path:
        return _error_html("Take a photo first."), None
    try:
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{API_URL}/scan/photo",
                files={"file": ("photo.jpg", f, "image/jpeg")},
                timeout=TIMEOUT,
            )
    except requests.RequestException as exc:
        return _error_html(f"Could not reach the API: {exc}"), None
    return _handle_response(r)


def scan_barcode(barcode: str):
    barcode = (barcode or "").strip()
    if not barcode:
        return _error_html("Enter a barcode first."), None
    if not barcode.isdigit():
        return _error_html("Barcodes are digits only (EAN-13 or UPC-A)."), None
    try:
        r = requests.post(
            f"{API_URL}/scan/barcode",
            json={"barcode": barcode},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return _error_html(f"Could not reach the API: {exc}"), None
    return _handle_response(r)


def _handle_response(r: requests.Response):
    if r.status_code == 404:
        return _error_html("Barcode not found in Open Food Facts."), None
    if r.status_code == 503:
        return _error_html("Model not loaded — train a checkpoint and restart."), None
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:
            pass
        return _error_html(f"API error {r.status_code}. {detail}"), None
    data = r.json()
    return _render_result(data), data


def _fmt(value, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:g} {unit}"


def _error_html(msg: str) -> str:
    return (f"<div style='padding:12px;border-left:4px solid #cf222e;"
            f"background:#fff1f1;border-radius:4px'>{msg}</div>")


def _render_result(data: dict) -> str:
    tier = data.get("confidence_tier") or "low"
    color, tier_label = TIER_STYLE.get(tier, TIER_STYLE["low"])
    conf = data.get("confidence")
    conf_txt = f"{conf:.0%}" if isinstance(conf, (int, float)) else "n/a"
    source = data.get("source") or "unknown"

    parts = [
        f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:8px'>"
        f"<span style='background:{color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:13px'>{tier_label}</span>"
        f"<span style='color:#666;font-size:13px'>confidence {conf_txt} · "
        f"source: {source}</span></div>"
    ]

    if data.get("food_unrecognized"):
        parts.append(
            "<p><b>Food not recognized.</b> Try a clearer photo or use the barcode tab.</p>"
        )
    else:
        name = data.get("name") or "Unknown"
        rows = "".join(
            f"<tr><td style='padding:4px 16px 4px 0'>{label}</td>"
            f"<td style='padding:4px 0;text-align:right'><b>{_fmt(data.get(key), unit)}</b></td></tr>"
            for key, label, unit in NUTRIENT_ROWS
        )
        parts.append(
            f"<h3 style='margin:4px 0'>{name}</h3>"
            f"<p style='color:#666;margin:0 0 8px;font-size:13px'>per 100 g</p>"
            f"<table style='border-collapse:collapse'>{rows}</table>"
        )

    candidates = data.get("top_3_candidates") or []
    if candidates:
        cand_rows = "".join(
            f"<tr><td style='padding:3px 16px 3px 0'>{c.get('label','?').replace('_',' ')}</td>"
            f"<td style='padding:3px 0;text-align:right'>{c.get('score',0):.0%}</td></tr>"
            for c in candidates
        )
        parts.append(
            "<h4 style='margin:12px 0 4px'>Did you mean…</h4>"
            f"<table style='border-collapse:collapse'>{cand_rows}</table>"
        )

    return "<div style='padding:4px'>" + "".join(parts) + "</div>"


with gr.Blocks(title="VitalScan Food Recognition") as demo:
    gr.Markdown("# VitalScan — Food Recognition")

    status = gr.Markdown(check_health())
    refresh = gr.Button("Refresh API status", size="sm")
    refresh.click(fn=check_health, outputs=status)

    with gr.Tab("📷 Scan food"):
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(
                    sources=["webcam", "upload"],
                    type="filepath",
                    label="Point camera at your food and capture",
                )
                scan_btn = gr.Button("Analyse food", variant="primary", size="lg")
            with gr.Column(scale=1):
                photo_result = gr.HTML(label="Result")
                with gr.Accordion("Raw JSON", open=False):
                    photo_json = gr.JSON()

        scan_btn.click(
            fn=lambda: (_loading_html(), None),
            outputs=[photo_result, photo_json],
            queue=False,
        ).then(
            fn=scan_photo,
            inputs=photo,
            outputs=[photo_result, photo_json],
        )

    with gr.Tab("🏷️ Barcode"):
        with gr.Row():
            with gr.Column(scale=1):
                barcode = gr.Textbox(label="Barcode (EAN-13 / UPC-A)",
                                     placeholder="e.g. 0049000028911")
                gr.Examples(
                    examples=[["0049000028911"], ["737628064502"], ["3017620422003"]],
                    inputs=barcode,
                    label="Try these",
                )
                barcode_btn = gr.Button("Look up", variant="primary")
            with gr.Column(scale=1):
                barcode_result = gr.HTML(label="Result")
                with gr.Accordion("Raw JSON", open=False):
                    barcode_json = gr.JSON()

        barcode_btn.click(
            fn=lambda: (_loading_html(), None),
            outputs=[barcode_result, barcode_json],
            queue=False,
        ).then(
            fn=scan_barcode,
            inputs=barcode,
            outputs=[barcode_result, barcode_json],
        )
        barcode.submit(fn=scan_barcode, inputs=barcode,
                       outputs=[barcode_result, barcode_json])


def _find_free_port(preferred: int | None = None) -> int:
    import socket
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = _find_free_port(int(port_env) if port_env else 7860)
    print(f"Launching VitalScan UI on http://127.0.0.1:{port}")
    demo.launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft())
