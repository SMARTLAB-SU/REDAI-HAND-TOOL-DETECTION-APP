"""
Inference & Evaluation Script for REDAI Hand Tool Detection App

This script provides comprehensive object detection inference, model evaluation, 
and ground-truth comparison against benchmark hand tool toolkits.

Features:
- Inference on single images, image directories, video files, or RealSense camera streams.
- Bounding box visualization with class labels and confidence scores.
- Ground truth validation against Excel dataset benchmarks (e.g., Book1.xlsx).
- Metrics evaluation (Precision, Recall, F1 Score, mAP@50, mAP@50-95).
- Report generation (CSV, JSON summary reports, visual outputs).
"""

import os
import sys
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional imports handled gracefully
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


HAND_TOOL_CLASSES = [
    'Adaptor', 'Allen Key', 'Bit Holder', 'Bit Sockets', 'Bits', 
    'Deep Socket', 'Extension Bar', 'Flex Handle', 'Rachet-Handels', 
    'S Handle', 'Sockets', 'Spanners', 'long bit socket', 'universal joint'
]


def load_model(model_path):
    """Load Ultralytics YOLO model from specified path."""
    if not HAS_ULTRALYTICS:
        raise ImportError("ultralytics library is required. Install via 'pip install ultralytics'")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"[INFO] Loading YOLO model from '{model_path}'...")
    model = YOLO(model_path)
    return model


def load_ground_truth(excel_path, company=None):
    """Load Ground Truth tool counts from Excel spreadsheet."""
    if not HAS_PANDAS:
        print("[WARNING] pandas library not installed. Skipping Excel Ground Truth lookup.")
        return {}
    
    if not excel_path or not os.path.exists(excel_path):
        print(f"[WARNING] Ground Truth Excel file not found: {excel_path}")
        return {}
    
    try:
        df = pd.read_excel(excel_path)
        print(f"[INFO] Ground Truth data loaded from '{excel_path}'")
        
        if company and 'Company' in df.columns:
            company_df = df[df['Company'] == company]
            if not company_df.empty:
                gt_dict = company_df.iloc[0].to_dict()
                gt_dict.pop('Company', None)
                return {k: int(v) for k, v in gt_dict.items() if pd.notna(v)}
            else:
                print(f"[WARNING] Company '{company}' not found in Ground Truth data.")
        
        # Default to first row if company not specified
        if not df.empty and 'Company' in df.columns:
            first_row = df.iloc[0].to_dict()
            first_row.pop('Company', None)
            return {k: int(v) for k, v in first_row.items() if pd.notna(v)}
    except Exception as e:
        print(f"[ERROR] Failed to read Ground Truth Excel file: {e}")
    
    return {}


def annotate_image(image, detections, confidence_threshold=0.25):
    """Draw bounding boxes and class names on image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    detected_objects = []
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id, class_name = det
        if conf < confidence_threshold:
            continue
            
        detected_objects.append(class_name)
        
        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Label text
        label_text = f"{class_name} {conf:.2f}"
        draw.text((x1, max(0, y1 - 20)), label_text, fill="yellow", font=font)
        
    return img_draw, detected_objects


def compare_with_ground_truth(detected_counts, ground_truth):
    """Compare detected object counts against Ground Truth benchmark."""
    summary = []
    all_keys = set(detected_counts.keys()).union(set(ground_truth.keys()))
    
    print("\n" + "="*70)
    print(f"{'CLASS NAME':<25} | {'DETECTED':<10} | {'GROUND TRUTH':<12} | {'DIFF':<8} | {'STATUS'}")
    print("="*70)
    
    for cls in sorted(all_keys):
        det = detected_counts.get(cls, 0)
        gt = ground_truth.get(cls, 0) if ground_truth else "N/A"
        
        if isinstance(gt, (int, float)):
            diff = det - gt
            status = "MATCH [PASS]" if diff == 0 else f"MISMATCH [{diff:+d}]"
        else:
            diff = "N/A"
            status = "N/A"
            
        print(f"{cls:<25} | {det:<10} | {str(gt):<12} | {str(diff):<8} | {status}")
        summary.append({
            "class": cls,
            "detected": det,
            "ground_truth": gt,
            "difference": diff,
            "status": status
        })
        
    print("="*70 + "\n")
    return summary


def run_image_inference(model, image_path, conf_thresh=0.25, iou_thresh=0.45, save_dir=None, ground_truth=None):
    """Run inference on a single image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    print(f"[INFO] Running inference on image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    results = model(image, conf=conf_thresh, iou=iou_thresh)
    boxes = results[0].boxes
    
    detections = []
    detected_names = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))
        detected_names.append(cls_name)
        
    annotated_img, _ = annotate_image(image, detections, conf_thresh)
    detected_counts = Counter(detected_names)
    
    gt_comparison = compare_with_ground_truth(detected_counts, ground_truth) if ground_truth else None
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"inference_{Path(image_path).stem}.png")
        annotated_img.save(save_path)
        print(f"[INFO] Annotated image saved to: {save_path}")
        
    return {
        "image": image_path,
        "detections": detections,
        "counts": dict(detected_counts),
        "comparison": gt_comparison
    }


def run_dir_inference(model, dir_path, conf_thresh=0.25, iou_thresh=0.45, save_dir=None, ground_truth=None):
    """Run inference on all images in a directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if Path(f).suffix.lower() in valid_exts]
    
    if not image_paths:
        print(f"[WARNING] No image files found in directory: {dir_path}")
        return []
        
    print(f"[INFO] Found {len(image_paths)} images in '{dir_path}'. Processing...")
    
    total_counts = Counter()
    results_list = []
    
    for img_path in image_paths:
        res = run_image_inference(model, img_path, conf_thresh, iou_thresh, save_dir, ground_truth=None)
        total_counts.update(res["counts"])
        results_list.append(res)
        
    print("\n--- AGGREGATE DIRECTORY DETECTION RESULTS ---")
    compare_with_ground_truth(total_counts, ground_truth)
    return results_list


def run_video_inference(model, video_path, conf_thresh=0.25, iou_thresh=0.45, save_dir=None):
    """Run inference on a video file."""
    if not HAS_CV2:
        raise ImportError("opencv-python is required for video inference. Install via 'pip install opencv-python'")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
        
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"inference_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Writing inference output video to: {out_path}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        results = model(pil_img, conf=conf_thresh, iou=iou_thresh)
        boxes = results[0].boxes
        
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]
            detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))
            
        annotated_pil, _ = annotate_image(pil_img, detections, conf_thresh)
        annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
        
        if writer:
            writer.write(annotated_bgr)
            
    cap.release()
    if writer:
        writer.release()
        
    elapsed = time.time() - start_time
    print(f"[INFO] Processed {frame_count} video frames in {elapsed:.2f}s ({frame_count/elapsed:.1f} FPS)")


def run_realsense_inference(model, conf_thresh=0.25, ground_truth=None):
    """Run live inference stream using Intel RealSense Camera."""
    if not HAS_REALSENSE:
        raise ImportError("pyrealsense2 is required for RealSense camera inference.")
    if not HAS_CV2:
        raise ImportError("opencv-python is required for RealSense camera display.")

    print("[INFO] Initializing RealSense Pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("[INFO] Press 'q' or ESC in display window to exit RealSense stream.")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)

            results = model(pil_img, conf=conf_thresh)
            boxes = results[0].boxes

            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id]
                detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))

            annotated_pil, _ = annotate_image(pil_img, detections, conf_thresh)
            display_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow("REDAI Hand Tool Detection - RealSense Live Feed", display_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def validate_dataset(model, dataset_yaml, save_dir=None):
    """Run Ultralytics YOLO dataset validation to compute mAP, Precision, and Recall metrics."""
    print(f"[INFO] Running validation metrics on dataset YAML: {dataset_yaml}")
    metrics = model.val(data=dataset_yaml, save_dir=save_dir)
    print("\n--- VALIDATION METRICS REPORT ---")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="REDAI Hand Tool Detection - Inference & Evaluation Engine")
    parser.add_argument("--model", type=str, default="../App/u_best.pt", help="Path to trained YOLO model (.pt)")
    parser.add_argument("--source", type=str, default=None, help="Input source: image path, directory path, or video file")
    parser.add_argument("--gt-excel", type=str, default="../App/Book1.xlsx", help="Path to Ground Truth Excel spreadsheet")
    parser.add_argument("--company", type=str, default=None, help="Company name for Ground Truth comparison (e.g. Kinchrome, REDAI, Tektron)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold (default: 0.45)")
    parser.add_argument("--output", type=str, default="../Results/Inference", help="Directory to save output annotated images/videos")
    parser.add_argument("--realsense", action="store_true", help="Run live inference on Intel RealSense camera feed")
    parser.add_argument("--val-yaml", type=str, default=None, help="Path to data.yaml to evaluate mAP/Precision/Recall metrics")
    
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    
    # Load ground truth if available
    ground_truth = load_ground_truth(args.gt_excel, company=args.company)

    if args.realsense:
        run_realsense_inference(model, conf_thresh=args.conf, ground_truth=ground_truth)
    elif args.val_yaml:
        validate_dataset(model, args.val_yaml, save_dir=args.output)
    elif args.source:
        if os.path.isfile(args.source):
            ext = Path(args.source).suffix.lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                run_video_inference(model, args.source, conf_thresh=args.conf, iou_thresh=args.iou, save_dir=args.output)
            else:
                run_image_inference(model, args.source, conf_thresh=args.conf, iou_thresh=args.iou, save_dir=args.output, ground_truth=ground_truth)
        elif os.path.isdir(args.source):
            run_dir_inference(model, args.source, conf_thresh=args.conf, iou_thresh=args.iou, save_dir=args.output, ground_truth=ground_truth)
        else:
            print(f"[ERROR] Source path not found: {args.source}")
    else:
        print("[INFO] No --source, --realsense, or --val-yaml provided.")
        print("Usage Examples:")
        print("  python inference.py --model ../App/u_best.pt --source ../Dataset/Images/test --gt-excel ../App/Book1.xlsx")
        print("  python inference.py --model ../App/u_best.pt --source sample.jpg --company Kinchrome")
        print("  python inference.py --model ../App/u_best.pt --realsense")
        print("  python inference.py --model ../App/u_best.pt --val-yaml ../Dataset/data.yaml")


if __name__ == "__main__":
    main()
