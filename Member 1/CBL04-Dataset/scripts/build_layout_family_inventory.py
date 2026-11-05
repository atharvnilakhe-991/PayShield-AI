"""Apply the reviewed visual-family rubric to the unique real-image inventory."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FAMILY_COMPACT = "Compact transaction details"
FAMILY_ENRICHED = "Merchant/context-enriched details"
FAMILY_TIMELINE = "Bank payment-progress timeline"
FAMILY_RECHARGE = "Recharge/utility detail summary"
RECHARGE_FROM_TIMELINE_CLUSTER = {
    "IMG-20261103-WA0321.jpg", "IMG-20261103-WA0459.jpg",
}
REASONS = {
    FAMILY_COMPACT: "Dark GPay completion summary followed by a compact bank/UPI details card and Google transaction ID.",
    FAMILY_ENRICHED: "Dark GPay transaction details with contextual payment/category or bank-confirmation content in the details card.",
    FAMILY_TIMELINE: "Dark GPay details card contains explicit payment-started/debited/sent/completed timeline stages.",
    FAMILY_RECHARGE: "Dark GPay recharge/utility summary with operator or service branding, plan/fee or bill fields, and transaction IDs.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    clusters = list(csv.DictReader((root / "reports/layout_visual_clusters.csv").open(encoding="utf-8-sig", newline="")))
    family_rows: list[dict[str, str]] = []
    for row in clusters:
        cluster = int(row["visual_cluster"])
        filename = row["filename"]
        if filename in RECHARGE_FROM_TIMELINE_CLUSTER:
            family = FAMILY_RECHARGE
        elif cluster == 4:
            family = FAMILY_COMPACT
        elif cluster in (1, 6):
            family = FAMILY_ENRICHED
        elif cluster == 5 or cluster == 3:
            family = FAMILY_TIMELINE if cluster == 5 or filename not in RECHARGE_FROM_TIMELINE_CLUSTER else FAMILY_RECHARGE
        elif cluster == 2:
            family = FAMILY_RECHARGE
        else:
            raise ValueError(f"Unexpected visual cluster {cluster} for {filename}")
        family_rows.append({"filename": filename, "family": family, "width": row["width"], "height": row["height"],
                            "visual_cluster": str(cluster), "reason": REASONS[family]})
    out = root / "reports/layout_family_inventory.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys())); writer.writeheader(); writer.writerows(family_rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in family_rows:
        grouped[row["family"]].append(row)
    lines = ["# Google Pay dark-theme layout family analysis", "",
             "## Method", "",
             "All 387 unique screenshots were reviewed in labeled contact sheets. A deterministic grayscale downsample cluster pass (12×36 pixels plus dimensions, k=6) was used only to surface structural groups; clusters were then visually reconciled into the four families below. Two recharge/utility screens fell into the bank-timeline visual cluster and were manually assigned to the recharge family. Counts refer to unique image contents; duplicate copies remain in raw storage and are excluded.", "",
             "## Families", ""]
    definitions = {
        FAMILY_COMPACT: ("Completion summary at the top (recipient/sender, amount, status, date/time), then a compact account/UPI detail card and Google transaction ID. Some have category or payment-method chips, but do not use the long progress timeline.", "Recipient or sender, amount, date/time, bank/account suffix, UPI transaction ID, counterparty UPI detail, Google transaction ID."),
        FAMILY_ENRICHED: ("Completion summary with contextual badges or a longer bank confirmation/message section in the details card. Includes merchant/merchant-like descriptions and receipt-confirmation text while retaining GPay branding.", "Merchant/counterparty, amount, completion message, bank/account, UPI transaction ID, sender/recipient, Google transaction ID."),
        FAMILY_TIMELINE: ("Expanded details card with sequential payment lifecycle events (such as payment started, amount debited, recipient credited, payment completed); some include the GPay bottom action strip.", "Counterparty, amount, event timestamps/statuses, bank, UPI transaction ID, sender/recipient, Google transaction ID."),
        FAMILY_RECHARGE: ("Operator/service-led receipt with a recharge or utility summary, plan/fee or bill breakdown, a progress/payment block, and transaction identifiers. Google Pay branding remains visible.", "Operator/service, plan or bill amount, fees/total, completion time, payer, mobile/account detail, UPI and Google transaction IDs."),
    }
    for name in (FAMILY_COMPACT, FAMILY_ENRICHED, FAMILY_TIMELINE, FAMILY_RECHARGE):
        members = grouped[name]
        widths = [int(r["width"]) for r in members]
        heights = [int(r["height"]) for r in members]
        reps = members[:3]
        lines.extend([f"### {name} — {len(members)} images", "",
                      definitions[name][0], "",
                      f"- Dimensions: width {min(widths)}–{max(widths)} px; height {min(heights)}–{max(heights)} px.",
                      f"- Representative files: {', '.join(r['filename'] for r in reps)}.",
                      f"- Key fields: {definitions[name][1]}", ""])
    lines.extend(["## Inventory totals", "", f"- Unique images classified: {len(family_rows)}.",
                  f"- Family counts: " + "; ".join(f"{name}: {len(grouped[name])}" for name in (FAMILY_COMPACT, FAMILY_ENRICHED, FAMILY_TIMELINE, FAMILY_RECHARGE)) + ".",
                  "- All families are Google Pay dark-theme screens; no light theme or other payment app was included.",
                  "- Seven redundant exact-duplicate copies are excluded from this unique-image count.", "",
                  "## Generator implications and limits", "",
                  "The first three families share a transaction-detail skeleton but differ in card content and vertical density. Recharge screens are a separate flow and need a distinct template. Real screenshots vary in width/height, crop, spacing, system chrome, bank logos, and bottom action controls. These four buckets capture the recurring visual families in this archive, not every individual app release or device variant. Synthetic output must use fictional names, account suffixes, and IDs, keep the GPay dark visual treatment, and carry a visible SYNTHETIC / DEMO mark.", ""])
    (root / "reports/layout_family_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print({name: len(grouped[name]) for name in (FAMILY_COMPACT, FAMILY_ENRICHED, FAMILY_TIMELINE, FAMILY_RECHARGE)})


if __name__ == "__main__":
    main()
