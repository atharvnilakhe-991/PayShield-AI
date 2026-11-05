from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model  # noqa: E402

SEED = 20261207
HEIGHT, WIDTH = 320, 160
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
EDITED_INDEX = 1
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
DEVICE = torch.device("cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


class ScreenshotDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(ROOT / row["image_path"]) as src:
            image = src.convert("RGB")
        with Image.open(ROOT / row["mask_path"]) as src:
            mask = src.convert("L")
        scale = min(WIDTH / image.width, HEIGHT / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        # BOX pooling followed by >0 threshold keeps a thin changed-pixel stroke
        # represented when original masks are downscaled to the network canvas.
        mask = mask.resize(size, Image.Resampling.BOX).point(lambda v: 255 if v > 0 else 0)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
        canvas.paste(image, (0, 0))
        mask_canvas = Image.new("L", (WIDTH, HEIGHT), 0)
        mask_canvas.paste(mask, (0, 0))
        rgb = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(HEIGHT, WIDTH, 3)
        x = rgb.permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        y = torch.frombuffer(bytearray(mask_canvas.tobytes()), dtype=torch.uint8).reshape(HEIGHT, WIDTH)
        y = (y > 127).float().unsqueeze(0)
        label = 0 if row["label"] == "clean" else 1
        return {"image": x, "mask": y, "label": torch.tensor(label, dtype=torch.long),
                "source_id": row["source_id"], "pair_id": row["pair_id"],
                "family": row["layout_family"], "edit_category": row["edit_category"],
                "image_path": row["image_path"]}


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def loss_fn(out: dict[str, torch.Tensor], labels: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls = F.cross_entropy(out["logits"], labels)
    bce = F.binary_cross_entropy_with_logits(out["mask_logits"], masks, pos_weight=torch.tensor(12.0, device=DEVICE))
    dice = dice_loss(out["mask_logits"], masks)
    seg = bce + dice
    return cls + .5 * seg, cls, seg


def classification_metrics(y: list[int], p: list[int]) -> dict[str, Any]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"support": len(y), "accuracy": (tp + tn) / max(1, len(y)), "precision": precision,
            "recall": recall, "f1": 2*precision*recall/max(1e-12, precision+recall),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def localization_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2*tp / max(1, 2*tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    return {"precision": precision, "recall": recall, "dice": dice, "iou": iou,
            "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn}


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = cls_total = seg_total = count = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            x, y, m = batch["image"].to(DEVICE), batch["label"].to(DEVICE), batch["mask"].to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, cls, seg = loss_fn(out, y, m)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = y.shape[0]
            total += loss.item() * bs; cls_total += cls.item() * bs; seg_total += seg.item() * bs; count += bs
    return {"loss": total/max(1,count), "classification_loss": cls_total/max(1,count),
            "localization_loss": seg_total/max(1,count)}


def infer(model: nn.Module, loader: DataLoader) -> tuple[list[dict[str, Any]], dict[str, int]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    pix: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(DEVICE))
            probs = torch.softmax(out["logits"], dim=1)[:, 1].cpu().tolist()
            mask_pred = torch.sigmoid(out["mask_logits"]) >= .5
            target = batch["mask"] >= .5
            for i, prob in enumerate(probs):
                true = int(batch["label"][i].item())
                pred = int(prob >= .5)
                a, b = mask_pred[i, 0], target[i, 0]
                tp = int((a & b).sum().item()); fp = int((a & ~b).sum().item()); fn = int((~a & b).sum().item())
                pix["tp"] += tp; pix["fp"] += fp; pix["fn"] += fn
                predictions.append({"source_id": batch["source_id"][i], "pair_id": batch["pair_id"][i],
                    "layout_family": batch["family"][i], "edit_category": batch["edit_category"][i],
                    "image_path": batch["image_path"][i], "true_label": true, "predicted_label": pred,
                    "edited_probability": prob, "mask_tp_pixels": tp, "mask_fp_pixels": fp, "mask_fn_pixels": fn})
    return predictions, pix


def draw_confusion(path: Path, matrix: dict[str, int], title: str = "Out-of-fold confusion matrix") -> None:
    im = Image.new("RGB", (540, 380), "white"); d = ImageDraw.Draw(im)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    d.text((120, 20), title, font=f, fill="black")
    d.text((310, 74), "Pred clean", font=f, fill="black"); d.text((410, 74), "Pred edited", font=f, fill="black")
    d.text((25, 150), "True clean", font=f, fill="black"); d.text((25, 250), "True edited", font=f, fill="black")
    for r, row in enumerate(((matrix["tn"], matrix["fp"]), (matrix["fn"], matrix["tp"]))):
        for c, value in enumerate(row):
            x, y = 280 + c*110, 120 + r*100
            d.rectangle((x,y,x+90,y+80), fill=(225,238,250), outline=(60,90,120), width=2)
            d.text((x+30,y+24), str(value), font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path)


def draw_bar(path: Path, title: str, values: list[tuple[str, float]], color: tuple[int,int,int]) -> None:
    width, height = 900, 480
    im = Image.new("RGB", (width,height), "white"); d=ImageDraw.Draw(im)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    d.text((20,15),title,font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",22),fill="black")
    left, top, chartw, charth = 190, 70, 660, 350
    for tick in range(0, 6):
        y = top + charth - tick*charth/5
        d.line((left,y,left+chartw,y),fill=(220,220,220),width=1)
        d.text((150,y-8),f"{tick/5:.1f}",font=font,fill=(60,60,60))
    gap = charth / max(1,len(values)); barh = gap*.55
    for i,(name,val) in enumerate(values):
        y=top+i*gap+gap*.2
        d.text((10,y+3),name,font=font,fill="black")
        d.rectangle((left,y,left+chartw*max(0,min(1,val)),y+barh),fill=color)
        d.text((left+chartw*max(0,min(1,val))+5,y+2),f"{val:.3f}",font=font,fill="black")
    path.parent.mkdir(parents=True,exist_ok=True); im.save(path)


def main() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg_path = ROOT / "templates/phase7/model_config.json"
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_cfg["architecture"]["pretrained"] = False
    runs = ROOT / "models/phase7"
    reports = ROOT / "reports/phase7"
    plots = reports / "plots"
    if runs.exists() and any(runs.iterdir()):
        raise FileExistsError(f"Model output already exists: {runs}; refusing to overwrite")
    runs.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    all_predictions: list[dict[str,Any]]=[]
    training_results=[]
    fold_localization={}
    started=time.time()
    for fold in range(1,5):
        random.seed(SEED+fold); torch.manual_seed(SEED+fold)
        fold_dir=ROOT/f"metadata/phase7/folds"
        train_rows=read_csv(fold_dir/f"fold_{fold}_train.csv")
        val_rows=read_csv(fold_dir/f"fold_{fold}_validation.csv")
        test_rows=read_csv(fold_dir/f"fold_{fold}_test.csv")
        train_loader=DataLoader(ScreenshotDataset(train_rows),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,
                                generator=torch.Generator().manual_seed(SEED+fold))
        val_loader=DataLoader(ScreenshotDataset(val_rows),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
        test_loader=DataLoader(ScreenshotDataset(test_rows),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
        model=build_model(model_cfg).to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        best_score=-1.0; best_state=None; history=[]
        for epoch in range(1,EPOCHS+1):
            train_metrics=run_epoch(model,train_loader,optimizer)
            val_metrics=run_epoch(model,val_loader,None)
            val_preds,_=infer(model,val_loader)
            vm=classification_metrics([x["true_label"] for x in val_preds],[x["predicted_label"] for x in val_preds])
            history.append({"epoch":epoch,"train":train_metrics,"validation":val_metrics,"validation_metrics":vm})
            print(json.dumps({"fold":fold,"epoch":epoch,"train_loss":train_metrics["loss"],
                              "val_loss":val_metrics["loss"],"val_f1":vm["f1"]}),flush=True)
            score=vm["f1"]
            if score>best_score:
                best_score=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"No checkpoint produced for fold {fold}")
        model.load_state_dict(best_state)
        checkpoint={"fold":fold,"seed":SEED+fold,"model_config":model_cfg,"state_dict":best_state,
                    "best_validation_f1":best_score,"training_history":history}
        torch.save(checkpoint,runs/f"fold_{fold}_resnet18_multitask.pt")
        preds,pixel_counts=infer(model,test_loader)
        for p in preds: p["fold"]=fold
        all_predictions.extend(preds)
        fold_metrics=classification_metrics([x["true_label"] for x in preds],[x["predicted_label"] for x in preds])
        fold_localization[str(fold)]=localization_from_counts(pixel_counts)
        training_results.append({"fold":fold,"train_pairs":len({r['pair_id'] for r in train_rows if r['label']=='clean'}),
                                 "validation_pairs":len({r['pair_id'] for r in val_rows if r['label']=='clean'}),
                                 "test_pairs":len({r['pair_id'] for r in test_rows if r['label']=='clean'}),
                                 "best_validation_f1":best_score,"best_epoch":1+max(range(len(history)),key=lambda i:history[i]["validation_metrics"]["f1"]),
                                 "test_metrics":fold_metrics,"test_localization":fold_localization[str(fold)],
                                 "history":history})
        print(json.dumps({"fold":fold,"best_val_f1":best_score,"test":fold_metrics}))

    master=classification_metrics([x["true_label"] for x in all_predictions],[x["predicted_label"] for x in all_predictions])
    global_pixels={k:sum(x[k] for x in [r for r in []]) for k in []}
    pixel_totals={"tp":sum(p["mask_tp_pixels"] for p in all_predictions),
                  "fp":sum(p["mask_fp_pixels"] for p in all_predictions),
                  "fn":sum(p["mask_fn_pixels"] for p in all_predictions)}
    localization=localization_from_counts(pixel_totals)
    by_family=defaultdict(list)
    for p in all_predictions: by_family[p["layout_family"]].append(p)
    family_metrics={family:classification_metrics([p["true_label"] for p in ps],[p["predicted_label"] for p in ps])
                    for family,ps in by_family.items()}
    family_local={family:localization_from_counts({k:sum(p[{"tp":"mask_tp_pixels","fp":"mask_fp_pixels","fn":"mask_fn_pixels"}[k]] for p in ps)
                                                   for k in ("tp","fp","fn")}) for family,ps in by_family.items()}
    category_metrics={}
    for category in sorted({p["edit_category"] for p in all_predictions if p["true_label"]==1}):
        positives=[p for p in all_predictions if p["true_label"]==1 and p["edit_category"]==category]
        fams={p["layout_family"] for p in positives}
        negatives=[p for p in all_predictions if p["true_label"]==0 and p["layout_family"] in fams]
        subset=negatives+positives
        category_metrics[category]=classification_metrics([p["true_label"] for p in subset],[p["predicted_label"] for p in subset])
        category_metrics[category]["edited_positive_support"]=len(positives)
        category_metrics[category]["clean_negative_support"]=len(negatives)
        category_metrics[category]["localization_on_edited"] = localization_from_counts({
            "tp":sum(p["mask_tp_pixels"] for p in positives),"fp":sum(p["mask_fp_pixels"] for p in positives),
            "fn":sum(p["mask_fn_pixels"] for p in positives)})
    false_positives=[p for p in all_predictions if p["true_label"]==0 and p["predicted_label"]==1]
    false_negatives=[p for p in all_predictions if p["true_label"]==1 and p["predicted_label"]==0]
    for p in false_positives+false_negatives:
        p["error_type"]="false_positive" if p["true_label"]==0 else "false_negative"
    write_csv(reports/"out_of_fold_predictions.csv",all_predictions)
    (reports/"training_history.json").write_text(json.dumps(training_results,indent=2),encoding="utf-8")
    (reports/"metrics.json").write_text(json.dumps({"classification":master,"localization":localization,
        "per_layout_family":family_metrics,"per_layout_localization":family_local,
        "per_edit_category":category_metrics,"per_fold":training_results,
        "false_positives":false_positives,"false_negatives":false_negatives,
        "synthetic_only":True,"real_ground_truth_used":False,"training_seconds":time.time()-started},indent=2),encoding="utf-8")
    training_config={"seed":SEED,"fold_seed_offset":1,"epochs":EPOCHS,"batch_size":BATCH_SIZE,"learning_rate":LR,
        "weight_decay":1e-4,"optimizer":"AdamW","device":"CPU","input_height":HEIGHT,"input_width":WIDTH,
        "resize":"aspect-preserving LANCZOS; top-left placement on dark canvas","mask_resize":"BOX then any-positive threshold",
        "classification_loss":"CrossEntropyLoss","localization_loss":"BCEWithLogits(pos_weight=12)+soft Dice",
        "loss_weights":{"classification":1.0,"localization":0.5},"checkpoint_selection":"highest validation edited-class F1",
        "pretrained":False,"split_strategy":"four-fold family-held-out LOFO","split_seed":20261206,
        "real_images_used":False,"training_pairs_per_fold":[228,272,228,272]}
    (ROOT/"metadata/phase7/training_config.json").write_text(json.dumps(training_config,indent=2),encoding="utf-8")
    draw_confusion(plots/"confusion_matrix.png",master["confusion_matrix"])
    draw_bar(plots/"per_layout_f1.png","Out-of-fold F1 by held-out layout family",[(k,v["f1"]) for k,v in family_metrics.items()],(51,125,190))
    draw_bar(plots/"per_edit_category_f1.png","Edit-category detection F1",[(k,v["f1"]) for k,v in category_metrics.items()],(54,150,90))
    draw_bar(plots/"localization_iou_by_family.png","Mask localization IoU by family",[(k,v["iou"]) for k,v in family_local.items()],(181,108,42))
    print(json.dumps({"pooled_oof":master,"localization":localization,"families":family_metrics,
                      "categories":category_metrics,"false_positives":len(false_positives),
                      "false_negatives":len(false_negatives),"seconds":time.time()-started},indent=2))


if __name__ == "__main__":
    main()
