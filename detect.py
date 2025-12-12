import argparse
import csv
import json
import os
import sys
from pathlib import Path
import pathlib
from datetime import datetime
import math

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import telepot 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import requests

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8292034134:AAHjiGjYGfivzW3IirfoyefhPFRKp3REAiw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1432479136")  

def telegram_api(method: str):
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        LOGGER.info("Telegram creds missing - skipping message.")
        return
    try:
        resp = requests.post(telegram_api("sendMessage"), data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
        if resp.status_code != 200:
            LOGGER.warning(f"Telegram message failed: {resp.status_code} {resp.text}")
    except Exception as e:
        LOGGER.warning(f"Telegram send error: {e}")

def send_telegram_file(file_path: Path, as_document: bool = False, caption: str = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        LOGGER.info("Telegram creds missing - skipping file send.")
        return
    method = "sendDocument" if as_document else "sendPhoto"
    url = telegram_api(method)
    try:
        with open(file_path, "rb") as f:
            files = {"document" if as_document else "photo": (file_path.name, f)}
            data = {"chat_id": TELEGRAM_CHAT_ID}
            if caption:
                data["caption"] = caption
            resp = requests.post(url, data=data, files=files, timeout=120)
            if resp.status_code != 200:
                LOGGER.warning(f"Telegram file send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        LOGGER.warning(f"Exception sending file to Telegram: {e}")


def severity_from_bbox(x1, y1, x2, y2, image_shape, confidence):
    """
    Compute severity by normalized bbox area and confidence.
    image_shape = (height, width, channels)
    Returns severity_str and numeric score (0..1).
    """
    ih, iw = image_shape[0], image_shape[1]
    box_area = max(0.0, (x2 - x1) * (y2 - y1))
    norm_area = box_area / (iw * ih + 1e-12)  # normalized area
    score = 0.6 * norm_area + 0.4 * confidence  # weights can be tuned
    # Map score to categories (thresholds can be tuned)
    if score < 0.01:
        sev = "Minor"
    elif score < 0.05:
        sev = "Moderate"
    else:
        sev = "Severe"
    score = min(max(score, 0.0), 1.0)
    return sev, float(score)

def solution_for_severity(severity: str):
    """Return recommended action string for a severity."""
    if severity == "Minor":
        return "Minor surface crack. Schedule visual monitoring and periodic recheck. No immediate closure required."
    elif severity == "Moderate":
        return "Moderate crack detected. Arrange on-site inspection and non-destructive testing (NDT). Prepare minor repairs."
    else:  # Severe
        return "Severe crack detected. Immediate structural inspection required. Limit access and consider emergency stabilization."

# ---------- Main run function ----------
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=True,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://","rtmp://","http://","https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    images_dir = save_dir / "images"
    graphs_dir = save_dir / "graphs"
    reports_dir = save_dir / "reports"
    csv_dir = save_dir / "csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "predictions.csv"

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    LOGGER.info(f"Loaded model names: {names}")

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # helper: write csv rows
    def append_csv_row(row):
        fieldnames = ["Image Name", "Prediction", "Confidence", "Severity", "Severity Score", "BBox", "Timestamp"]
        file_exists = csv_path.is_file()
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize_flag = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize_flag).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize_flag).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize_flag)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_image_path = images_dir / p.name
            txt_path = str(save_dir / "labels" / (p.stem + ("" if dataset.mode == "image" else f"_{frame}")))
            s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1,0,1,0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            timestamp = datetime.now().isoformat()
            per_image_detections = []  # for graph/report
            per_image_rows = []

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # count per class for this image
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s'*(n>1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if not hide_conf else names[c]
                    confidence = float(conf)
                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                    severity, sev_score = severity_from_bbox(x1, y1, x2, y2, im0.shape, confidence)
                    solution = solution_for_severity(severity)
                    bbox_str = f"[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"

                    # CSV/JSON row
                    row = {
                        "Image Name": p.name,
                        "Prediction": label,
                        "Confidence": f"{confidence:.3f}",
                        "Severity": severity,
                        "Severity Score": f"{sev_score:.4f}",
                        "BBox": bbox_str,
                        "Timestamp": timestamp
                    }
                    per_image_rows.append(row)
                    if save_csv:
                        try:
                            append_csv_row(row)
                        except Exception as e:
                            LOGGER.warning(f"Failed writing CSV row: {e}")

                    # Annotate
                    disp_label = None if hide_labels else f"{label} {confidence:.2f} ({severity})"
                    annotator.box_label(xyxy, disp_label, color=colors(c, True))

                    # Save crop
                    if save_crop:
                        crop_dir = save_dir / "crops" / label
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        save_one_box(xyxy, imc, file=crop_dir / f"{p.stem}.jpg", BGR=True)

                    per_image_detections.append({
                        "class_id": c,
                        "class_name": label,
                        "confidence": float(confidence),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "severity": severity,
                        "severity_score": sev_score,
                        "recommended_action": solution
                    })

            # Final annotated image
            annotated_img = annotator.result()
            # Save annotated image (always save if save_img True)
            if save_img:
                try:
                    cv2.imwrite(str(save_image_path), annotated_img)
                except Exception as e:
                    LOGGER.warning(f"Failed to save annotated image: {e}")

            # Create per-image graph (detection counts + confidences)
            graph_path = None
            if len(per_image_detections) > 0:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(10,4))
                    # counts by class
                    cls_names = [d["class_name"] for d in per_image_detections]
                    uniq, counts = np.unique(cls_names, return_counts=True)
                    axes[0].bar(uniq, counts)
                    axes[0].set_title("Detections by Class")
                    axes[0].tick_params(axis='x', rotation=45)
                    # confidences histogram
                    confs = [d["confidence"] for d in per_image_detections]
                    axes[1].hist(confs, bins=8)
                    axes[1].set_title("Confidence Distribution")
                    fig.tight_layout()
                    graph_path = graphs_dir / f"{p.stem}_analysis.png"
                    fig.savefig(graph_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    LOGGER.warning(f"Failed to create graph: {e}")
                    graph_path = None

            # Create per-image JSON report
            report_path = reports_dir / f"{p.stem}_report.json"
            try:
                report = {
                    "image_name": p.name,
                    "timestamp": timestamp,
                    "detections": per_image_detections,
                    "summary": {
                        "total_detections": len(per_image_detections),
                        "generated_at": timestamp
                    }
                }
                with open(report_path, "w") as fh:
                    json.dump(report, fh, indent=2)
            except Exception as e:
                LOGGER.warning(f"Failed writing JSON report: {e}")
                report_path = None

            # Create PDF report (if available)
            pdf_path = None
            if REPORTLAB_AVAILABLE:
                try:
                    pdf_path = reports_dir / f"{p.stem}_report.pdf"
                    c = canvas.Canvas(str(pdf_path), pagesize=A4)
                    W, H = A4
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(50, H-60, f" Defect Report - {p.name}")
                    c.setFont("Helvetica", 10)
                    c.drawString(50, H-80, f"Generated: {timestamp}")
                    c.drawString(50, H-95, f"Total detections: {len(per_image_detections)}")
                    # annotated image
                    try:
                        # scale image to fit
                        img_w = 480
                        img_h = int(img_w * annotated_img.shape[0] / annotated_img.shape[1])
                        tmp_img_path = reports_dir / f"{p.stem}_annot_temp.jpg"
                        cv2.imwrite(str(tmp_img_path), annotated_img)
                        c.drawImage(str(tmp_img_path), 50, H-120-img_h, width=img_w, height=img_h)
                        try:
                            tmp_img_path.unlink()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # small table of detections
                    y = H - 140 - img_h if 'img_h' in locals() else H-140
                    c.setFont("Helvetica", 9)
                    for det_info in per_image_detections[:10]:  # limit rows on PDF to 10
                        y -= 14
                        text = f"{det_info['class_name']} conf:{det_info['confidence']:.2f} sev:{det_info['severity']} action:{det_info['recommended_action']}"
                        c.drawString(50, y, text[:120])
                    c.showPage()
                    c.save()
                except Exception as e:
                    LOGGER.warning(f"Failed to create PDF: {e}")
                    pdf_path = None

            # Send Telegram alert if any detections or always (you choose)
            if len(per_image_detections) > 0:
                # build human-friendly message
                total = len(per_image_detections)
                classes_here = {}
                max_conf = 0.0
                sev_counts = {}
                for d in per_image_detections:
                    classes_here[d["class_name"]] = classes_here.get(d["class_name"], 0) + 1
                    max_conf = max(max_conf, d["confidence"])
                    sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1

                msg = f"🚨 * Defect Alert* 🚨\n"
                msg += f"Image: `{p.name}`\n"
                msg += f"Time: {timestamp}\n"
                msg += f"Total detections: {total}\n"
                msg += "By class:\n"
                for k, v in classes_here.items():
                    msg += f"  • {k}: {v}\n"
                msg += "By severity:\n"
                for k, v in sev_counts.items():
                    msg += f"  • {k}: {v}\n"
                msg += f"Max confidence: {max_conf:.2f}\n\n"
                # suggestions summary (top-most severe)
                if "Severe" in sev_counts:
                    msg += "⚠️ *Immediate action recommended*: There is at least one *Severe* detection.\n"
                    msg += "Recommended: " + solution_for_severity("Severe") + "\n"
                elif "Moderate" in sev_counts:
                    msg += "⚠️ *Moderate issues found* — schedule inspection.\n"
                    msg += "Recommended: " + solution_for_severity("Moderate") + "\n"
                else:
                    msg += "Info: Minor issues detected; recommend monitoring.\n"
                    msg += "Recommended: " + solution_for_severity("Minor") + "\n"

                # send text (Markdown-friendly, use parse_mode if desired)
                try:
                    send_telegram_message(msg)
                except Exception as e:
                    LOGGER.warning(f"Telegram message error: {e}")

                # send annotated image
                if save_img and save_image_path.exists():
                    try:
                        send_telegram_file(save_image_path, as_document=False, caption="Annotated detection")
                    except Exception as e:
                        LOGGER.warning(f"Send annotated image failed: {e}")

                # send graph
                if graph_path and graph_path.exists():
                    try:
                        send_telegram_file(graph_path, as_document=False, caption="Detection analysis graph")
                    except Exception as e:
                        LOGGER.warning(f"Send graph failed: {e}")

                # send PDF
                if pdf_path and pdf_path.exists():
                    try:
                        send_telegram_file(pdf_path, as_document=True, caption="PDF Detection Report")
                    except Exception as e:
                        LOGGER.warning(f"Send PDF failed: {e}")

            # end per-image loop

            # Show image if requested
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

    # End dataset loop

    # Final summary and overall graphs
    # (Minimal summary here — you can extend to aggregate across run)
    final_summary_path = save_dir / "run_summary.txt"
    with open(final_summary_path, "w") as fh:
        fh.write(f"Processed images: {seen}\n")
    try:
        send_telegram_message(f"✅ Processing complete. Images processed: {seen}.\nReports in: {save_dir}")
        # optionally send run summary image if created
    except Exception:
        pass

    LOGGER.info(f"Done. Results saved to {save_dir}")

# ---------- CLI ----------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "dam.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=2, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    try:
        check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    except Exception:
        pass
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
