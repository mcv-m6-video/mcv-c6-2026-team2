#!/usr/bin/env python3
import argparse
import torch
import os
import io
import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Imports de tu proyecto
from util.io import load_json
from util.eval_spotting import apply_NMS 
from dataset.datasets import get_datasets
from model import get_model
from main_spotting import update_args
 
OUT_WIDTH = 800
HUD_HEIGHT = 80   
PLOT_HEIGHT = 220
PAUSE_FRAMES = 5  
SCORE_THRESH = 0.05 
NMS_WINDOW = 5    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--num_random', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='qualitative_results')
    parser.add_argument('--show_chart', action='store_true')
    parser.add_argument('--soft', action='store_true')
    return parser.parse_args()

def get_top_preds(probs, idx_to_class, top_k=2, threshold=SCORE_THRESH):
    action_probs = probs[1:] 
    top_indices = np.argsort(action_probs)[::-1][:top_k]
    preds = []
    for i in top_indices:
        if action_probs[i] > threshold:
            preds.append(f"{idx_to_class[i + 1]} ({action_probs[i]:.2f})")
    return " | ".join(preds) if preds else "Background"

def build_clean_plot(probs_raw, probs_nms, gt_labels, clip_len, classes):
    dpi = 100
    fig = plt.figure(figsize=(OUT_WIDTH / dpi, PLOT_HEIGHT / dpi), dpi=dpi)
    ax = fig.add_axes([0.1, 0.25, 0.85, 0.65])
    
    t_axis = np.arange(clip_len)
    idx_to_class = {v: k for k, v in classes.items()}
    
    # Graficamos SOLO lo que hay en el GT para máxima limpieza
    present_classes = np.unique(gt_labels)
    colors = plt.cm.tab10.colors

    for i, c_idx in enumerate(sorted(present_classes)):
        if c_idx == 0: continue 
        color = colors[i % 10]
        class_name = idx_to_class[c_idx]
        
        ax.plot(t_axis, probs_raw[:, c_idx], color=color, alpha=0.4, linewidth=2, label=f"{class_name}")
        
        # Diamantes NMS
        nms_hits = np.where(probs_nms[:, c_idx-1] > 0.1)[0]
        for hit_frame in nms_hits:
            ax.scatter(hit_frame, probs_nms[hit_frame, c_idx-1], color=color, s=70, marker='D', edgecolor='white', zorder=5)

        # GT (Línea roja)
        for f in np.where(gt_labels == c_idx)[0]:
            ax.axvline(x=f, color='red', linestyle='--', alpha=0.7)
            ax.text(f, 1.02, 'GT', color='red', fontsize=8, fontweight='bold', ha='center')

    ax.set_xlim(0, clip_len - 1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Conf.", fontsize=9)
    ax.grid(True, alpha=0.1, linestyle='--')
    ax.legend(loc='upper right', fontsize=8)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    plt.close(fig)
    return cv2.resize(img, (OUT_WIDTH, PLOT_HEIGHT))

def render_expert_gif(frames, out_path, gt_labels, probs_raw, probs_nms, classes, show_chart=True):
    T, C, H, W = frames.shape
    idx_to_class = {v: k for k, v in classes.items()}
    scaled_h = int(H * OUT_WIDTH / W)
    total_h = HUD_HEIGHT + scaled_h + (PLOT_HEIGHT if show_chart else 0)
    
    gif_frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    plot_base = build_clean_plot(probs_raw, probs_nms, gt_labels, T, classes) if show_chart else None

    # Texto estático de GT
    gt_list = [f"{idx_to_class[gt_labels[t]]}@{t}" for t in range(T) if gt_labels[t] != 0]
    gt_summary = "GT: " + (", ".join(gt_list) if gt_list else "None")

    for t in range(T):
        canvas = np.zeros((total_h, OUT_WIDTH, 3), dtype=np.uint8)
        
        # Video
        img_rgb = frames[t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_bgr = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (OUT_WIDTH, scaled_h))
        canvas[HUD_HEIGHT : HUD_HEIGHT + scaled_h, :] = img_bgr

        # HUD 
        # Izquierda: Info básica y GT
        cv2.putText(canvas, f"F: {t}", (15, 25), font, 0.5, (150, 150, 150), 1)
        cv2.putText(canvas, gt_summary, (15, 60), font, 0.6, (0, 255, 0), 2)
        
        # Derecha: Predicciones actuales (Top 2)
        top_preds = get_top_preds(probs_raw[t], idx_to_class)
        cv2.putText(canvas, f"PRED: {top_preds}", (OUT_WIDTH // 2, 60), font, 0.6, (0, 220, 255), 2)

        # Plot
        if show_chart and plot_base is not None:
            plot_frame = plot_base.copy()
            playhead_x = int(OUT_WIDTH * 0.1) + int((t / (T - 1)) * (OUT_WIDTH * 0.85))
            cv2.line(plot_frame, (playhead_x, 0), (playhead_x, PLOT_HEIGHT), (0, 0, 0), 2)
            canvas[HUD_HEIGHT + scaled_h : , :] = plot_frame

        # Pausa con Overlay Gris
        current_nms = probs_nms[t]
        is_spot = np.max(current_nms) > 0
        is_gt = gt_labels[t] != 0

        if is_spot or is_gt:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (200, HUD_HEIGHT+40), (600, HUD_HEIGHT+130), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
            
            msg = f"GT: {idx_to_class[gt_labels[t]]}" if is_gt else f"PRED: {idx_to_class[np.argmax(current_nms)+1]}"
            color = (0, 255, 0) if is_gt else (0, 220, 255)
            cv2.putText(canvas, msg, (230, HUD_HEIGHT+95), font, 1.1, color, 3)

        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if is_spot or is_gt:
            for _ in range(PAUSE_FRAMES): gif_frames.append(gif_frames[-1])

    imageio.mimsave(out_path, gif_frames, fps=6, loop=0)

def main():
    args = get_args()
    config = load_json(f"config/{args.model}.json")
    args = update_args(args, config)
    classes, _, _, dataset = get_datasets(args)
    idx_to_class = {v: k for k, v in classes.items()}

    model = get_model(args=args)
    model._model.load_state_dict(torch.load(args.checkpoint, map_location=model.device))
    model._model.eval()

    indices = args.indices if args.indices else np.random.choice(len(dataset), args.num_random).tolist()
    
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    for idx in indices:
        sample = dataset[idx]
        gt_labels = sample["label"].numpy()
        if args.soft:
            gt_labels = np.argmax(gt_labels, axis=1)
        
        # FILTRO DE CAOS 
        num_eventos = np.count_nonzero(gt_labels)
        if num_eventos > 7 or num_eventos == 0:
            print(f"Saltando clip {idx}: tiene {num_eventos} eventos (buscamos 1 o 2).")
            continue
        
        probs_raw, _ = model.predict(sample["frame"])
        probs_raw = probs_raw[0]
        probs_nms = apply_NMS(probs_raw[:, 1:].copy(), window=NMS_WINDOW, thresh=SCORE_THRESH)
        
        event_names = list(set([idx_to_class[l] for l in gt_labels if l != 0]))
        save_path = os.path.join(out_dir, f"clip_{idx}_{'_'.join(event_names)}.gif")

        print(f"Generando: {save_path}")
        render_expert_gif(sample["frame"], save_path, gt_labels, probs_raw, probs_nms, classes, show_chart=args.show_chart)

if __name__ == "__main__":
    main()