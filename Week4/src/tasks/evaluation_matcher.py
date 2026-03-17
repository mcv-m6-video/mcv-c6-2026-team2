import logging
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.matcher import get_matcher, initialize_matcher
from src.utils.camera import Camera, compute_relationships
from src.utils.car import Car
from src.utils.track_manager import TrackManager


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configures console logging for matcher evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_sequence_metadata(dataset_root: str, seq: str):
    """Loads cameras, timestamps, frame counts, videos, and GT detections for a sequence."""
    root = Path(dataset_root)
    seq_root = root / "train" / seq
    LOGGER.info("Loading sequence metadata from %s", seq_root)

    # Camera timestamps let us align each video on a shared global timeline.
    offsets = {}
    with open(root / "cam_timestamp" / f"{seq}.txt", "r") as f:
        for line in f:
            cam_name, offset = line.strip().split()
            offsets[cam_name] = float(offset)

    # These frame counts are used to stop reading each camera at the right time.
    num_frames = {}
    with open(root / "cam_framenum" / f"{seq}.txt", "r") as f:
        for line in f:
            cam_name, n_frames = line.strip().split()
            num_frames[cam_name] = int(n_frames)

    cameras = []
    gt_by_camera = []
    video_paths = []
    camera_names = []

    for cam_idx, cam_dir in enumerate(sorted(seq_root.glob("c*"))):
        camera_names.append(cam_dir.name)
        video_paths.append(str(cam_dir / "vdo.avi"))
        # GT is grouped by local frame so later we can ask "which cars are visible now?"
        gt_by_camera.append(load_gt_detections(cam_dir / "gt" / "gt.txt"))

        # Calibration is needed to project detections into GPS space and reason
        # about overlapping/adjacent cameras, exactly like in the matching task.
        with open(cam_dir / "calibration.txt", "r") as f:
            homography_str = f.readline().strip().split(" ")[2:]
            homography_matrix = [[]]
            for element in homography_str:
                subelements = element.split(";")
                homography_matrix[-1].append(subelements[0])
                if len(subelements) > 1:
                    homography_matrix.append([subelements[1]])
            homography = np.array(homography_matrix, dtype=np.float32)

        cap = cv2.VideoCapture(str(cam_dir / "vdo.avi"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        cameras.append(
            Camera(
                cam_idx,
                (width, height),
                homography,
                offsets[cam_dir.name],
                num_frames[cam_dir.name],
            )
        )

    # Precompute which cameras overlap and which ones are plausible transitions.
    cameras = compute_relationships(cameras)
    LOGGER.info(
        "Loaded %d cameras for %s: %s",
        len(cameras),
        seq,
        camera_names,
    )
    return cameras, gt_by_camera, video_paths, camera_names


def load_gt_detections(gt_path: Path):
    """Loads GT detections grouped by local frame index."""
    frame_to_dets = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            # Store boxes as xyxy because that is the format used by the rest
            # of the matching code.
            x2 = x + w
            y2 = y + h
            frame_to_dets[frame_idx].append(
                {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "bbox": np.array([x, y, x2, y2], dtype=np.float32),
                }
            )
    return frame_to_dets


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Creates a deterministic BGR color from an id."""
    rng = np.random.default_rng(track_id + 12345)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def annotate_frame(frame: np.ndarray, detections: list[dict], cam_idx: int, t_manager: TrackManager):
    """Draws predicted global ids and bounding boxes on a frame."""
    annotated = frame.copy()
    for det in detections:
        local_id = det["track_id"]
        bbox = det["bbox"].astype(int)
        key = (cam_idx, local_id)
        global_id = t_manager.local_to_global.get(key, -1)
        color = color_for_id(global_id if global_id >= 0 else local_id)
        x1, y1, x2, y2 = bbox.tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"G{global_id} | L{local_id}"
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def collect_tracks_from_gt(dataset_root: str, seq: str):
    """Builds one Car instance per GT track using all detections in the sequence."""
    cameras, gt_by_camera, video_paths, camera_names = load_sequence_metadata(
        dataset_root, seq)
    track_manager = TrackManager()
    local_cars_registry: dict[int, dict[int, Car]] = {
        i: {} for i in range(len(cameras))}

    for cam_idx, (video_path, gt_frames, camera) in enumerate(zip(video_paths, gt_by_camera, cameras)):
        LOGGER.info("Collecting GT tracks for camera %s",
                    camera_names[cam_idx])
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = gt_frames.get(frame_idx, [])
            for det in detections:
                local_id = det["track_id"]
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox.astype(int)
                car_image = frame[y1:y2, x1:x2]

                if local_id not in local_cars_registry[cam_idx]:
                    local_cars_registry[cam_idx][local_id] = Car(local_id)

                car = local_cars_registry[cam_idx][local_id]
                car.add_detection(
                    car_image,
                    bbox,
                    camera.homography,
                    frame_idx,
                    cam_idx,
                    confidence=1.0,  # GT detections have perfect confidence
                )
                track_manager.register_car(cam_idx, local_id, car)

            if frame_idx == 1 or frame_idx % 500 == 0:
                LOGGER.info(
                    "Collected camera %s | frame %d | active_local_tracks=%d",
                    camera_names[cam_idx],
                    frame_idx,
                    len(local_cars_registry[cam_idx]),
                )

            frame_idx += 1

        cap.release()
        LOGGER.info(
            "Finished camera %s | local_tracks=%d",
            camera_names[cam_idx],
            len(local_cars_registry[cam_idx]),
        )

    return cameras, gt_by_camera, video_paths, camera_names, track_manager, local_cars_registry


def match_all_tracks(track_manager: TrackManager, local_cars_registry: dict[int, dict[int, Car]]):
    """Compares every local track against every other local track."""
    tracks = [
        (cam_idx, local_id, car)
        for cam_idx, cars in local_cars_registry.items()
        for local_id, car in cars.items()
    ]
    LOGGER.info(
        "Starting all-to-all track comparison over %d tracks", len(tracks))

    num_pairs = 0
    num_matches = 0
    pairwise_decisions = []
    matcher = get_matcher()
    if matcher is None:
        raise RuntimeError(
            "Matcher must be initialized before comparing tracks.")

    for idx, (cam_idx_a, local_id_a, car_a) in enumerate(tracks):
        for cam_idx_b, local_id_b, car_b in tracks[idx + 1:]:
            if cam_idx_a == cam_idx_b:
                continue

            num_pairs += 1
            true_same = local_id_a == local_id_b
            similarity = matcher.similarity(car_a.embeddings, car_b.embeddings)
            pred_same = similarity >= matcher.similarity_threshold

            pairwise_decisions.append(
                {
                    "cam_idx_a": cam_idx_a,
                    "local_id_a": local_id_a,
                    "cam_idx_b": cam_idx_b,
                    "local_id_b": local_id_b,
                    "true_same": true_same,
                    "similarity": similarity,
                    "pred_same": pred_same,
                }
            )

            if pred_same:
                track_manager.link_cars(
                    cam_idx_a, local_id_a, cam_idx_b, local_id_b)
                num_matches += 1

        if (idx + 1) % 25 == 0 or idx == len(tracks) - 1:
            LOGGER.info(
                "Compared %d/%d tracks | checked_pairs=%d | accepted_matches=%d",
                idx + 1,
                len(tracks),
                num_pairs,
                num_matches,
            )

    LOGGER.info("Finished all-to-all matching | checked_pairs=%d | accepted_matches=%d",
                num_pairs, num_matches)
    return pairwise_decisions


def compute_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Computes standard binary classification metrics from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn +
                            tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_random_baseline_metrics(
    pairwise_decisions: list[dict],
    predict_same_probability: float,
) -> dict:
    """Computes the expected metrics of a random matcher.

    Args:
        pairwise_decisions: Ground-truth pair annotations.
        predict_same_probability: Probability of predicting "same".

    Returns:
        Expected confusion counts and metrics for the random baseline.
    """
    num_pairs = len(pairwise_decisions)
    if num_pairs == 0:
        return {
            "predict_same_probability": predict_same_probability,
            "num_compared_pairs": 0,
            **compute_binary_metrics(0, 0, 0, 0),
        }

    num_positive = sum(
        1 for decision in pairwise_decisions if decision["true_same"])
    num_negative = num_pairs - num_positive

    tp = predict_same_probability * num_positive
    fn = (1.0 - predict_same_probability) * num_positive
    fp = predict_same_probability * num_negative
    tn = (1.0 - predict_same_probability) * num_negative

    metrics = compute_binary_metrics(tp, fp, fn, tn)
    metrics["predict_same_probability"] = predict_same_probability
    metrics["num_compared_pairs"] = num_pairs
    return metrics


def compute_pair_prevalence(pairwise_decisions: list[dict]) -> float:
    """Computes the fraction of positive pairs in the evaluated set."""
    if not pairwise_decisions:
        return 0.0
    return sum(1 for decision in pairwise_decisions if decision["true_same"]) / len(pairwise_decisions)


def compute_pairwise_decision_metrics(pairwise_decisions: list[dict]) -> dict:
    """Scores the raw one-to-one matcher decisions before any transitive merging."""
    tp = fp = fn = tn = 0
    for decision in pairwise_decisions:
        true_same = decision["true_same"]
        pred_same = decision["pred_same"]

        if true_same and pred_same:
            tp += 1
        elif not true_same and pred_same:
            fp += 1
        elif true_same and not pred_same:
            fn += 1
        else:
            tn += 1

    metrics = compute_binary_metrics(tp, fp, fn, tn)
    metrics["num_compared_pairs"] = len(pairwise_decisions)
    return metrics


def compute_pairwise_decision_metrics_at_threshold(
    pairwise_decisions: list[dict],
    threshold: float,
) -> dict:
    """Scores raw matcher decisions after thresholding similarities."""
    thresholded = []
    for decision in pairwise_decisions:
        thresholded.append(
            {
                **decision,
                "pred_same": decision["similarity"] >= threshold,
            }
        )

    metrics = compute_pairwise_decision_metrics(thresholded)
    metrics["threshold"] = threshold
    return metrics


def compute_pairwise_metrics(track_manager: TrackManager):
    """Computes pairwise metrics over the final merged global assignments."""
    items = []
    for (cam_idx, local_id), pred_global_id in track_manager.local_to_global.items():
        items.append(
            {
                "cam_idx": cam_idx,
                "local_id": local_id,
                # With GT detections, the track id is our true cross-camera identity.
                "true_id": local_id,
                "pred_id": pred_global_id,
            }
        )

    tp = fp = fn = tn = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            # We score clustering quality pairwise: should these two camera-local
            # tracks be merged or kept apart?
            same_true = items[i]["true_id"] == items[j]["true_id"]
            same_pred = items[i]["pred_id"] == items[j]["pred_id"]

            if same_true and same_pred:
                tp += 1
            elif not same_true and same_pred:
                fp += 1
            elif same_true and not same_pred:
                fn += 1
            else:
                tn += 1

    metrics = {
        "num_camera_local_tracks": len(items),
        "num_true_global_ids": len({item["true_id"] for item in items}),
        "num_pred_global_ids": len({item["pred_id"] for item in items}),
    }
    metrics.update(compute_binary_metrics(tp, fp, fn, tn))
    return metrics


def build_track_manager_for_threshold(
    local_cars_registry: dict[int, dict[int, Car]],
    pairwise_decisions: list[dict],
    threshold: float,
) -> TrackManager:
    """Builds final global assignments by merging all pairs above a threshold."""
    track_manager = TrackManager()
    for cam_idx, cars in local_cars_registry.items():
        for local_id, car in cars.items():
            track_manager.register_car(cam_idx, local_id, car)

    accepted_matches = 0
    for decision in pairwise_decisions:
        if decision["similarity"] < threshold:
            continue

        track_manager.link_cars(
            decision["cam_idx_a"],
            decision["local_id_a"],
            decision["cam_idx_b"],
            decision["local_id_b"],
        )
        accepted_matches += 1

    LOGGER.info(
        "Applied threshold %.3f | accepted pairwise matches=%d",
        threshold,
        accepted_matches,
    )
    return track_manager


def build_thresholds(start: float, end: float, step: float) -> list[float]:
    """Builds an inclusive list of thresholds."""
    if step <= 0:
        raise ValueError("threshold_step must be > 0")
    thresholds = []
    current = start
    while current <= end + 1e-9:
        thresholds.append(round(current, 6))
        current += step
    return thresholds


def plot_precision_recall_curve(
    metrics_by_threshold: list[dict],
    pairwise_decisions: list[dict],
    output_path: Path,
) -> None:
    """Plots a precision-recall curve from threshold sweep results."""

    recalls = [m["pairwise_decision_metrics"]["recall"]
               for m in metrics_by_threshold]
    precisions = [m["pairwise_decision_metrics"]["precision"]
                  for m in metrics_by_threshold]
    thresholds = [m["threshold"] for m in metrics_by_threshold]
    fair_coin = metrics_by_threshold[0]["random_baselines"]["fair_coin"] if metrics_by_threshold else None
    prevalence_matched = (
        metrics_by_threshold[0]["random_baselines"]["prevalence_matched"]
        if metrics_by_threshold
        else None
    )
    random_sweep = []
    if pairwise_decisions:
        for prob in np.arange(0.0, 1.0001, 0.1):
            random_sweep.append(
                compute_random_baseline_metrics(
                    pairwise_decisions, float(round(prob, 2)))
            )

    plt.figure(figsize=(9, 7))

    # --- Main curve ---
    sns.lineplot(x=recalls, y=precisions, marker="o", label="Matcher sweep")

    # Annotate thresholds
    for r, p, t in zip(recalls, precisions, thresholds):
        plt.text(r, p, f"{t:.2f}", fontsize=12, ha="left", va="bottom")

    # --- Random sweep (if exists) ---
    if random_sweep:
        random_recalls = [m["recall"] for m in random_sweep]
        random_precisions = [m["precision"] for m in random_sweep]

        sns.lineplot(
            x=random_recalls,
            y=random_precisions,
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Random baseline sweep",
        )

        for m in random_sweep:
            plt.text(
                m["recall"],
                m["precision"],
                f"{m['predict_same_probability']:.1f}",
                fontsize=12,
                color="gray",
            )

    # --- Baseline points ---
    if fair_coin is not None:
        plt.scatter(
            fair_coin["recall"],
            fair_coin["precision"],
            marker="X",
            s=100,
            label="Random baseline: fair coin (50%)",
            color="purple",
        )

    if prevalence_matched is not None:
        plt.scatter(
            prevalence_matched["recall"],
            prevalence_matched["precision"],
            marker="^",
            s=100,
            label="Random baseline:\nprevalence-matched (5.6%)",
            color="orange",
        )

    # --- Axis limits ---
    plt.xlim(0, 1.05)
    plt.ylim(0, 1)

    # --- Custom grid every 0.1 ---
    ticks = np.arange(0, 1.01, 0.1)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # --- Labels ---
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Matcher Pairwise Precision-Recall Curve")

    plt.legend(
        title="Matcher labels are thresholds;\ngray labels are random positive rates",
        loc="best",
        title_fontsize="13",
        fontsize="13",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info("Saved precision-recall curve to %s", output_path)


def render_annotated_videos(
    output_dir: str,
    seq: str,
    video_paths: list[str],
    gt_by_camera: list[dict],
    camera_names: list[str],
    assignments: dict[tuple[int, int], int],
):
    """Renders final annotated videos using the finished global assignments."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Rendering annotated videos to %s", output_root)

    for cam_idx, (video_path, gt_frames, cam_name) in enumerate(
        zip(video_paths, gt_by_camera, camera_names)
    ):
        # This second pass is important: we render with the final global ids
        # after all cross-camera merges have already been decided.
        LOGGER.info("Rendering camera %s from %s", cam_name, video_path)
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        writer = cv2.VideoWriter(
            str(output_root / f"{seq}_{cam_name}_matched.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = gt_frames.get(frame_idx, [])
            annotated = frame.copy()
            for det in detections:
                local_id = det["track_id"]
                bbox = det["bbox"].astype(int)
                # Each local GT track receives the predicted global id assigned
                # by the matcher pipeline.
                global_id = assignments.get((cam_idx, local_id), -1)
                color = color_for_id(global_id if global_id >= 0 else local_id)
                x1, y1, x2, y2 = bbox.tolist()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"G{global_id} | L{local_id}",
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated)
            if frame_idx == 1 or frame_idx % 500 == 0:
                LOGGER.info(
                    "Rendered camera %s | frame %d",
                    cam_name,
                    frame_idx,
                )
            frame_idx += 1

        cap.release()
        writer.release()
        LOGGER.info("Finished rendering camera %s", cam_name)


def run_matcher_on_gt(dataset_root: str, seq: str):
    """Runs offline all-to-all matching using GT detections."""
    LOGGER.info("Starting GT-based matcher evaluation on sequence %s", seq)
    cameras, gt_by_camera, video_paths, camera_names, track_manager, local_cars_registry = collect_tracks_from_gt(
        dataset_root, seq
    )

    # After all local tracks are built, compare each one against all the others.
    pairwise_decisions = match_all_tracks(track_manager, local_cars_registry)
    positive_prevalence = compute_pair_prevalence(pairwise_decisions)

    decision_metrics = compute_pairwise_decision_metrics(pairwise_decisions)
    clustering_metrics = compute_pairwise_metrics(track_manager)
    metrics = {
        "pairwise_positive_prevalence": positive_prevalence,
        "random_baselines": {
            "fair_coin": compute_random_baseline_metrics(pairwise_decisions, 0.5),
            "prevalence_matched": compute_random_baseline_metrics(pairwise_decisions, positive_prevalence),
        },
        "pairwise_decision_metrics": decision_metrics,
        "final_clustering_metrics": clustering_metrics,
    }
    LOGGER.info("Finished matcher inference on %s", seq)
    LOGGER.info("Pairwise decision metrics: %s", decision_metrics)
    LOGGER.info("Final clustering metrics: %s", clustering_metrics)
    return (
        metrics,
        pairwise_decisions,
        local_cars_registry,
        track_manager.local_to_global,
        gt_by_camera,
        video_paths,
        camera_names,
    )


def main(args):
    setup_logging()
    dataset_root = args.dataset_root
    seq = args.seq
    match_checkpoint = args.match_checkpoint
    output_dir = getattr(args, "eval_output_dir",
                         os.path.join("outputs", "matcher_eval", seq))
    thresholds = build_thresholds(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    LOGGER.info(
        "Evaluation config | dataset_root=%s | seq=%s | checkpoint=%s | output_dir=%s | thresholds=%s",
        dataset_root,
        seq,
        match_checkpoint,
        output_dir,
        thresholds,
    )

    initialize_matcher(match_checkpoint)
    if get_matcher() is None:
        raise RuntimeError("Matcher could not be initialized for evaluation.")
    LOGGER.info("Matcher initialized successfully")

    # First pass: collect tracks and similarity scores once.
    default_metrics, pairwise_decisions, local_cars_registry, assignments, gt_by_camera, video_paths, camera_names = run_matcher_on_gt(
        dataset_root, seq
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics_by_threshold = []
    best_metrics = None
    best_assignments = assignments

    for threshold in thresholds:
        pairwise_metrics = compute_pairwise_decision_metrics_at_threshold(
            pairwise_decisions, threshold
        )
        threshold_track_manager = build_track_manager_for_threshold(
            local_cars_registry,
            pairwise_decisions,
            threshold,
        )
        clustering_metrics = compute_pairwise_metrics(threshold_track_manager)
        threshold_metrics = {
            "threshold": threshold,
            "pairwise_positive_prevalence": compute_pair_prevalence(pairwise_decisions),
            "random_baselines": {
                "fair_coin": compute_random_baseline_metrics(pairwise_decisions, 0.5),
                "prevalence_matched": compute_random_baseline_metrics(
                    pairwise_decisions,
                    compute_pair_prevalence(pairwise_decisions),
                ),
            },
            "pairwise_decision_metrics": pairwise_metrics,
            "final_clustering_metrics": clustering_metrics,
            "beats_fair_coin_f1": pairwise_metrics["f1"] > compute_random_baseline_metrics(pairwise_decisions, 0.5)["f1"],
            "beats_prevalence_matched_f1": pairwise_metrics["f1"] > compute_random_baseline_metrics(
                pairwise_decisions,
                compute_pair_prevalence(pairwise_decisions),
            )["f1"],
        }
        metrics_by_threshold.append(threshold_metrics)

        if best_metrics is None or pairwise_metrics["f1"] > best_metrics["pairwise_decision_metrics"]["f1"]:
            best_metrics = threshold_metrics
            best_assignments = threshold_track_manager.local_to_global.copy()

    summary_metrics = {
        "default_matcher_threshold_metrics": default_metrics,
        "threshold_sweep": metrics_by_threshold,
        "best_threshold_by_pairwise_f1": best_metrics,
    }

    render_threshold = args.render_threshold if args.render_threshold is not None else best_metrics[
        "threshold"]
    LOGGER.info("Rendering videos with threshold %.3f", render_threshold)
    render_track_manager = build_track_manager_for_threshold(
        local_cars_registry,
        pairwise_decisions,
        render_threshold,
    )
    # Second pass: render the videos using the chosen predicted assignments.
    render_annotated_videos(
        output_dir=output_dir,
        seq=seq,
        video_paths=video_paths,
        gt_by_camera=gt_by_camera,
        camera_names=camera_names,
        assignments=render_track_manager.local_to_global,
    )
    metrics_path = Path(output_dir) / f"{seq}_metrics.json"
    assignments_path = Path(output_dir) / f"{seq}_assignments.json"
    pairwise_decisions_path = Path(
        output_dir) / f"{seq}_pairwise_decisions.json"
    pr_curve_path = Path(output_dir) / f"{seq}_precision_recall_curve.png"

    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    LOGGER.info("Saved metrics to %s", metrics_path)

    serializable_assignments = {
        f"{cam_idx}:{local_id}": global_id
        for (cam_idx, local_id), global_id in render_track_manager.local_to_global.items()
    }
    with open(assignments_path, "w") as f:
        json.dump(serializable_assignments, f, indent=2)
    LOGGER.info("Saved assignments to %s", assignments_path)

    with open(pairwise_decisions_path, "w") as f:
        json.dump(pairwise_decisions, f, indent=2)
    LOGGER.info("Saved pairwise decisions to %s", pairwise_decisions_path)
    plot_precision_recall_curve(
        metrics_by_threshold, pairwise_decisions, pr_curve_path)

    print(f"Saved annotated videos and metrics to {output_dir}")
    print(json.dumps(summary_metrics, indent=2))
