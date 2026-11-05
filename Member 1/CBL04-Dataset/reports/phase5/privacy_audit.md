# Phase 5 real-data privacy audit

## Finding: treat the entire real archive as sensitive

The 387 unique files are Google Pay dark-theme payment screenshots. Their visible fields include sender/recipient or merchant names, dates and amounts, bank names and account suffixes, masked phone or UPI details, and UPI/Google transaction reference identifiers. Masking part of a phone or account value does not make a screenshot safe to publish; transaction references and names can still identify people or link to financial activity. The screenshots may also expose sensitive merchant, timing, and spending information.

The archive contains 394 readable JPEG files, including seven exact duplicate copies. Duplicate copies do not reduce the privacy risk. No personal names, account values, IDs, or real-image thumbnails are reproduced in this report or the Phase 5 contact sheets.

## Handling recommendation

- Keep originals in access-controlled internal storage; do not publish, email broadly, or place in public repositories, demos, papers, or issue trackers.
- Use the synthetic set for public examples and routine development.
- For any internal ML use of real screenshots, confirm an appropriate lawful basis and data-owner approval, restrict access and retention, and assess whether the task can be done without the real set.
- If derived de-identified examples are later authorized, create a separate derivative copy, irreversibly redact names, phone/UPI/account details, transaction references, timestamps and other linkable fields, remove metadata, and independently inspect the result. Keep the originals unchanged and access-restricted; do not assume reversible blur or partial masking is sufficient.
- Do not include real screenshots or OCR text in future contact sheets or reports.

This audit records sensitive field types and handling risk only; it does not copy or disclose the values.
