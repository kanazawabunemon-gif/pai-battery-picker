"""Train the waste-tray detector (YOLO-OBB) on the auto-labelled synthetic dataset.

  python perception/train_yolo.py --data in/obb_ds --task obb --epochs 80 --out out/yolo_obb
Writes out/yolo/best.pt, metrics.json (mAP per class) and a few prediction images.
"""
import argparse
import json
import os
import shutil

TRASH_NAMES = ["battery", "powerbank", "can", "pen", "cap", "paper", "snackbox", "carton"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default="obb", choices=["obb"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    CLASS_NAMES = TRASH_NAMES
    a.model = a.model or "yolo11s-obb.pt"

    data = os.path.abspath(a.data)
    yaml_path = os.path.join(data, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {data}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    a.out = os.path.abspath(a.out)      # Ultralytics puts relative projects under ./runs/
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.model)
    model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.out, name="run",
                hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
                degrees=180, translate=0.1, scale=0.3, fliplr=0.5, flipud=0.5,
                mosaic=1.0, close_mosaic=10, patience=30, plots=True, verbose=False)
    run = os.path.join(a.out, "run")
    best = os.path.join(run, "weights", "best.pt")
    shutil.copyfile(best, os.path.join(a.out, "best.pt"))
    m = YOLO(best).val(data=yaml_path, imgsz=a.imgsz, split="val", plots=False, verbose=False)
    b = m.box
    metrics = {"task": a.task, "model": a.model, "mAP50": float(b.map50), "mAP50_95": float(b.map),
               "per_class_mAP50_95": {CLASS_NAMES[i]: float(v) for i, v in enumerate(b.maps)},
               "per_class_recall": {CLASS_NAMES[i]: float(b.r[j]) for j, i in enumerate(b.ap_class_index)},
               "per_class_precision": {CLASS_NAMES[i]: float(b.p[j]) for j, i in enumerate(b.ap_class_index)},
               "precision": float(b.mp), "recall": float(b.mr)}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    # prediction samples for the report
    val_imgs = sorted(os.listdir(os.path.join(data, "images", "val")))[:8]
    res = YOLO(best).predict([os.path.join(data, "images", "val", v) for v in val_imgs], imgsz=a.imgsz,
                             conf=0.4, verbose=False)
    os.makedirs(os.path.join(a.out, "pred"), exist_ok=True)
    for r, v in zip(res, val_imgs):
        r.save(filename=os.path.join(a.out, "pred", v))


if __name__ == "__main__":
    main()
