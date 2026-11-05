# Google Pay dark-theme layout family analysis

## Method

All 387 unique screenshots were reviewed in labeled contact sheets. A deterministic grayscale downsample cluster pass (12×36 pixels plus dimensions, k=6) was used only to surface structural groups; clusters were then visually reconciled into the four families below. Two recharge/utility screens fell into the bank-timeline visual cluster and were manually assigned to the recharge family. Counts refer to unique image contents; duplicate copies remain in raw storage and are excluded.

## Families

### Compact transaction details — 169 images

Completion summary at the top (recipient/sender, amount, status, date/time), then a compact account/UPI detail card and Google transaction ID. Some have category or payment-method chips, but do not use the long progress timeline.

- Dimensions: width 540–732 px; height 1319–1600 px.
- Representative files: IMG-20261103-WA0014.jpg, IMG-20261103-WA0018.jpg, IMG-20261103-WA0021.jpg.
- Key fields: Recipient or sender, amount, date/time, bank/account suffix, UPI transaction ID, counterparty UPI detail, Google transaction ID.

### Merchant/context-enriched details — 167 images

Completion summary with contextual badges or a longer bank confirmation/message section in the details card. Includes merchant/merchant-like descriptions and receipt-confirmation text while retaining GPay branding.

- Dimensions: width 420–738 px; height 1280–1601 px.
- Representative files: IMG-20261103-WA0017.jpg, IMG-20261103-WA0020.jpg, IMG-20261103-WA0022.jpg.
- Key fields: Merchant/counterparty, amount, completion message, bank/account, UPI transaction ID, sender/recipient, Google transaction ID.

### Bank payment-progress timeline — 42 images

Expanded details card with sequential payment lifecycle events (such as payment started, amount debited, recipient credited, payment completed); some include the GPay bottom action strip.

- Dimensions: width 353–478 px; height 1280–1600 px.
- Representative files: IMG-20261103-WA0205.jpg, IMG-20261103-WA0210.jpg, IMG-20261103-WA0217.jpg.
- Key fields: Counterparty, amount, event timestamps/statuses, bank, UPI transaction ID, sender/recipient, Google transaction ID.

### Recharge/utility detail summary — 9 images

Operator/service-led receipt with a recharge or utility summary, plan/fee or bill breakdown, a progress/payment block, and transaction identifiers. Google Pay branding remains visible.

- Dimensions: width 270–471 px; height 1248–1603 px.
- Representative files: IMG-20261103-WA0138.jpg, IMG-20261103-WA0216.jpg, IMG-20261103-WA0220.jpg.
- Key fields: Operator/service, plan or bill amount, fees/total, completion time, payer, mobile/account detail, UPI and Google transaction IDs.

## Inventory totals

- Unique images classified: 387.
- Family counts: Compact transaction details: 169; Merchant/context-enriched details: 167; Bank payment-progress timeline: 42; Recharge/utility detail summary: 9.
- All families are Google Pay dark-theme screens; no light theme or other payment app was included.
- Seven redundant exact-duplicate copies are excluded from this unique-image count.

## Generator implications and limits

The first three families share a transaction-detail skeleton but differ in card content and vertical density. Recharge screens are a separate flow and need a distinct template. Real screenshots vary in width/height, crop, spacing, system chrome, bank logos, and bottom action controls. These four buckets capture the recurring visual families in this archive, not every individual app release or device variant. Synthetic output must use fictional names, account suffixes, and IDs, keep the GPay dark visual treatment, and carry a visible SYNTHETIC / DEMO mark.
