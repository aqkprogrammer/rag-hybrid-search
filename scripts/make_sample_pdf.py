# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab>=4.2"]
# ///
"""Regenerate `sample_docs/it/laptop-and-byod-standard.pdf` (a real, text-based PDF).

Run with:  uv run scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "it" / "laptop-and-byod-standard.pdf"

PAGES: list[list[tuple[str, str]]] = [
    [
        ("h1", "Laptop and BYOD Standard"),
        ("p", "Owner: IT Operations. Applies to all Lumora Systems employees and contractors."),
        ("h2", "Standard Laptop Issue"),
        (
            "p",
            "Every new employee receives a company-managed laptop on or before their first day. "
            "Engineers receive a MacBook Pro with 36 GB of memory; all other roles receive a "
            "MacBook Air with 16 GB of memory. Windows laptops are available on request for "
            "roles that require them.",
        ),
        (
            "p",
            "Laptops are refreshed every 3 years. Employees may request an early replacement if "
            "the device fails the hardware diagnostic or can no longer receive security updates.",
        ),
        ("h2", "Device Management"),
        (
            "p",
            "All company laptops are enrolled in Kandji mobile device management (MDM) before "
            "they are shipped. Removing or disabling the MDM profile is a violation of the "
            "Information Security Policy and is reported to the Security team automatically.",
        ),
    ],
    [
        ("h2", "Bring Your Own Device (BYOD)"),
        (
            "p",
            "Personal phones and tablets may be used to access email, calendar and Slack only "
            "after they are enrolled in the Lumora BYOD program through the Intune Company "
            "Portal app. Enrolled personal devices must run a supported operating system version "
            "and use a passcode of at least 6 digits.",
        ),
        (
            "p",
            "Personal laptops and desktop computers must never be used to access internal "
            "systems, source code or customer data.",
        ),
        ("h2", "Lost or Stolen Devices"),
        (
            "p",
            "Report a lost or stolen laptop or enrolled phone to IT through the #it-help channel "
            "immediately, and in any case within 1 hour. IT will lock and remotely wipe the "
            "device. A replacement laptop is shipped within 2 business days.",
        ),
        ("h2", "Peripherals"),
        (
            "p",
            "Employees may order one external monitor, a keyboard, a mouse and a headset through "
            "the IT Service Portal. Additional peripherals can be purchased with the home office "
            "stipend.",
        ),
    ],
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, page in enumerate(PAGES):
        for kind, text in page:
            style = {"h1": styles["Title"], "h2": styles["Heading2"], "p": styles["BodyText"]}[kind]
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 6))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        title="Laptop and BYOD Standard",
        author="IT Operations",
        subject="IT",
    )
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
