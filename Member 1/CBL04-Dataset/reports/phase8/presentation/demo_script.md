# Hackathon demo script (2–3 minutes)

## Before presenting

1. From `CBL04-Dataset`, activate `.venv_phase7` and run `python app/server.py`.
2. Open `http://127.0.0.1:8080`.
3. Prepare one fictional sample from each family under `data/synthetic_gpay_phase3/clean/` or `edited/`; keep the visible SYNTHETIC / DEMO marker in view.
4. Do not use real payment screenshots or the held-out Phase 7 test set.

## Talk track

“We’re demonstrating a research prototype that studies deliberate field changes in synthetic Google Pay-style dark transaction screens. It does not verify a payment or determine fraud.”

“I’ll select a fictional, visibly marked demo screenshot. The application keeps the selected preview in the browser and sends the image to a local CPU model. The output is only the model’s experimental class guess and an uncalibrated score.”

Select one clean and one edited synthetic sample from different layout families. Explain that the model may be wrong and that output is not a probability. Point out that localization is disabled because the measured masks were unreliable.

“Our partial synthetic-only test accuracy was 50.5%. One completed fold labeled every compact screenshot edited, and another missed most edited enriched examples. Two layout families still lack held-out test results. We have not demonstrated real-world payment fraud performance.”

“The next work is to complete the remaining family-held-out folds, increase independent layouts, and test paired-reference or validated forensic approaches. For a genuine transaction, confirm it in the bank or payment app.”

## If asked

- **Does it prove a screenshot is fake?** No. It is a weak experimental model.
- **Is the score a probability?** No. It is an uncalibrated softmax output.
- **Can it highlight the changed text?** The prototype disables highlights because localization accuracy was poor.
- **Were the 387 real screenshots labeled clean?** No. They remain unlabeled and were not used as clean controls.
- **Why still demonstrate it?** To show the data, model-integration workflow, and current limitations honestly.
