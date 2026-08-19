"""Create a short multi-page handbook used to demo PDF citations."""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pdfs" / "sample_acme_handbook.pdf"

PAGES = [
    (
        "ACME Employee Handbook - Page 1",
        "ACME Corp remote-work policy\n\n"
        "Effective 1 March 2025, full-time employees may work remotely up to three days "
        "per week. Core collaboration hours are 11:00 to 16:00 in the employee's local "
        "timezone. Managers may grant a fully remote arrangement after a 90-day review.",
    ),
    (
        "ACME Employee Handbook - Page 2",
        "Leave and expense policy\n\n"
        "Annual leave is 22 days. Parental leave is 16 weeks at full pay. "
        "Travel expenses over 75 USD require pre-approval in the finance portal. "
        "The security team must be notified before any customer data leaves ACME laptops.",
    ),
    (
        "ACME Employee Handbook - Page 3",
        "On-call and incident response\n\n"
        "The primary on-call engineer acknowledges P1 incidents within 15 minutes. "
        "A written incident report is due within 48 hours. Access to production "
        "databases is limited to Staff Engineers and the on-call rotation.",
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        page.insert_textbox(pymupdf.Rect(72, 120, 540, 720), body, fontsize=12, align=0)
    doc.save(OUT)
    doc.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
