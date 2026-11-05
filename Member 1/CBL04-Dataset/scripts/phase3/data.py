from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any


NAMES = ["Aarav Mehta", "Isha Kulkarni", "Kabir Sharma", "Mira Nair", "Rohan Desai",
         "Anaya Joshi", "Dev Malhotra", "Tara Iyer", "Neel Patil", "Sara Rao"]
SERVICES = ["Demo Mobile Service", "Sample Transit Top-Up", "Fictional Utility Service", "Demo FastTag Service"]
BANKS = ["Sample Bank", "Demo National Bank", "Fictional Co-op Bank", "Test Payments Bank"]
CONTEXTS = ["Groceries", "Cafe payment", "Bill payment", "Store purchase", "Travel"]


def example(rng: random.Random, family: str) -> dict[str, Any]:
    recipient = rng.choice(SERVICES) if family == "recharge" else rng.choice(NAMES)
    amount = round(rng.uniform(90, 2400), 2) if family == "recharge" else rng.choice([125, 280, 450, 999, 1250, 2400])
    dt = datetime(2025, 1, 1) + timedelta(days=rng.randrange(700), minutes=rng.randrange(1440))
    return {
        "recipient": recipient,
        "sender": rng.choice(NAMES),
        "amount": amount,
        "plan_price": round(amount - 1.0, 2) if family == "recharge" else amount,
        "service_fee": 1.0 if family == "recharge" else 0.0,
        "date_time": f"{dt.day:02d} {dt.strftime('%b %Y')}, {dt.hour % 12 or 12}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}",
        "bank": rng.choice(BANKS),
        "account_last": f"{rng.randrange(1000, 10000)}",
        "phone_last": f"{rng.randrange(10, 100)}",
        "recipient_upi": f"••••{rng.randrange(1000,10000)}@demo · Google Pay",
        "upi_id": f"DEMO-UPI-{rng.randrange(10**7,10**8)}",
        "google_id": f"DEMO-GPAY-{rng.randrange(10**7,10**8)}-{rng.randrange(100,1000)}",
        "context": rng.choice(CONTEXTS),
    }


def edit_value(source: dict[str, Any], field: str, rng: random.Random) -> tuple[dict[str, Any], str, str]:
    edited = dict(source)
    old = str(source["upi_id"] if field == "transaction_id" else source["account_last"] if field == "other" else source[field])
    if field == "amount":
        edited[field] = round(float(source[field]) + rng.choice([37, 61, 105]), 2) if isinstance(source[field], float) else source[field] + rng.choice([37, 61, 105])
        if source["plan_price"] == source["amount"] - source["service_fee"]:
            edited["plan_price"] = round(edited["amount"] - source["service_fee"], 2)
    elif field == "recipient":
        options = SERVICES if source["recipient"] in SERVICES else NAMES
        edited[field] = rng.choice([name for name in options if name != source[field]])
    elif field == "date_time":
        dt = datetime.strptime(source[field], "%d %b %Y, %I:%M%p") + timedelta(hours=2)
        edited[field] = f"{dt.day:02d} {dt.strftime('%b %Y')}, {dt.hour % 12 or 12}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"
    elif field == "transaction_id":
        edited["upi_id"] = f"DEMO-UPI-{rng.randrange(10**7,10**8)}"
        old = source["upi_id"]
    elif field == "other":
        edited["account_last"] = f"{(int(source['account_last']) + rng.randrange(1, 9000)) % 10000:04d}"
        old = source["account_last"]
    else:
        raise ValueError(f"Unsupported edit field: {field}")
    new = str(edited["upi_id"] if field == "transaction_id" else edited["account_last"] if field == "other" else edited[field])
    return edited, old, new
