import logging
import json
import os
from collections import defaultdict
from pathlib import Path
import re

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.matcher import get_matcher, initialize_matcher
from src.utils.camera import Camera, compute_relationships
from src.utils.car import Car
from src.utils.reid_training_dataset import ReIDFolderDataset, split_indices_by_pid
from src.utils.render_video import render_assignment_videos
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
        roi_mask = cv2.imread(str(cam_dir / "roi.jpg"), cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            raise FileNotFoundError(
                f"ROI mask not found for camera {cam_dir.name}: {cam_dir / 'roi.jpg'}")

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
                roi_mask,
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


def resolve_split_track_filters(
    dataset_root: str,
    sequences: list[str],
    train_sequences_arg: str,
    val_sequences_arg: str,
    val_ratio: float,
    split_seed: int,
    split_subset: str,
) -> dict[str, set[int] | None]:
    """Rebuilds the train/val split for one or more sequences from CLI args."""
    train_sequences = [
        seq_name.strip()
        for seq_name in train_sequences_arg.split(",")
        if seq_name.strip()
    ]
    val_sequences = [
        seq_name.strip()
        for seq_name in val_sequences_arg.split(",")
        if seq_name.strip()
    ]

    if split_subset == "all":
        LOGGER.info(
            "Evaluating full sequences without split filtering: %s", sequences)
        return {seq: None for seq in sequences}

    if not train_sequences and not val_sequences:
        raise ValueError(
            "split_subset requires CLI split information. Provide --train_sequences and, if applicable, --val_sequences."
        )

    if val_sequences:
        if split_subset == "train":
            raise ValueError(
                "split_subset='train' is not supported when the checkpoint used dedicated validation sequences."
            )
        invalid_sequences = [
            seq for seq in sequences if seq not in val_sequences]
        if invalid_sequences:
            raise ValueError(
                f"Sequences {invalid_sequences} are not part of the dedicated validation sequences provided on the command line: {val_sequences}"
            )
        LOGGER.info(
            "Checkpoint used dedicated validation sequences. Evaluating the full validation sequences %s.",
            sequences,
        )
        return {seq: None for seq in sequences}

    invalid_sequences = [
        seq for seq in sequences if seq not in train_sequences]
    if invalid_sequences:
        raise ValueError(
            f"Sequences {invalid_sequences} are not part of the training sequences provided on the command line: {train_sequences}"
        )

    base_train_set = ReIDFolderDataset(
        dataset_root,
        sequences=train_sequences,
        transform=None,
    )
    train_indices, val_indices = split_indices_by_pid(
        base_train_set.samples,
        val_ratio=val_ratio,
        seed=split_seed,
    )

    selected_indices = val_indices if split_subset == "val" else train_indices

    allowed_track_ids_by_sequence = {seq: set() for seq in sequences}
    for idx in selected_indices:
        sample = base_train_set.samples[idx]
        if sample.sequence not in allowed_track_ids_by_sequence:
            continue
        seq_num = int(re.sub(r"\D", "", sample.sequence))
        seq_prefix = seq_num * 1_000_000
        allowed_track_ids_by_sequence[sample.sequence].add(
            sample.pid - seq_prefix)

    for seq in sequences:
        LOGGER.info(
            "Rebuilt %s split from CLI args | seq=%s | val_ratio=%.3f | seed=%d | kept_tracks=%d",
            split_subset,
            seq,
            val_ratio,
            split_seed,
            len(allowed_track_ids_by_sequence[seq]),
        )
    return allowed_track_ids_by_sequence


def collect_tracks_from_gt(
    dataset_root: str,
    seq: str,
    allowed_track_ids: set[int] | None = None,
):
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
                if allowed_track_ids is not None and local_id not in allowed_track_ids:
                    continue
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


def match_all_tracks(
    track_manager: TrackManager,
    local_cars_registry: dict[int, dict[int, Car]],
    seq: str,
):
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
                    "sequence": seq,
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


def aggregate_clustering_metrics(metrics_list: list[dict]) -> dict:
    """Aggregates per-sequence clustering metrics into one combined report."""
    tp = sum(metrics["true_positive"] for metrics in metrics_list)
    fp = sum(metrics["false_positive"] for metrics in metrics_list)
    fn = sum(metrics["false_negative"] for metrics in metrics_list)
    tn = sum(metrics["true_negative"] for metrics in metrics_list)

    aggregated = {
        "num_camera_local_tracks": sum(metrics["num_camera_local_tracks"] for metrics in metrics_list),
        "num_true_global_ids": sum(metrics["num_true_global_ids"] for metrics in metrics_list),
        "num_pred_global_ids": sum(metrics["num_pred_global_ids"] for metrics in metrics_list),
    }
    aggregated.update(compute_binary_metrics(tp, fp, fn, tn))
    return aggregated


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
    plot_title: str,
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
    plt.title(plot_title)

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


def run_matcher_on_gt(
    dataset_root: str,
    seq: str,
    allowed_track_ids: set[int] | None = None,
):
    """Runs offline all-to-all matching using GT detections."""
    LOGGER.info("Starting GT-based matcher evaluation on sequence %s", seq)
    cameras, gt_by_camera, video_paths, camera_names, track_manager, local_cars_registry = collect_tracks_from_gt(
        dataset_root,
        seq,
        allowed_track_ids=allowed_track_ids,
    )

    # After all local tracks are built, compare each one against all the others.
    pairwise_decisions = match_all_tracks(
        track_manager, local_cars_registry, seq)
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
        allowed_track_ids,
    )


def main(args):
    setup_logging()
    dataset_root = args.dataset_root
    sequences = [seq.strip() for seq in args.seq.split(",") if seq.strip()]
    match_checkpoint = args.match_checkpoint
    split_subset = getattr(args, "split_subset", "all")
    seq_label = "_".join(sequences)
    output_dir = getattr(args, "eval_output_dir",
                         os.path.join("outputs", "matcher_eval", seq_label))
    thresholds = build_thresholds(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    LOGGER.info(
        "Evaluation config | dataset_root=%s | seqs=%s | checkpoint=%s | split_subset=%s | output_dir=%s | thresholds=%s",
        dataset_root,
        sequences,
        match_checkpoint,
        split_subset,
        output_dir,
        thresholds,
    )

    initialize_matcher(match_checkpoint)
    if get_matcher() is None:
        raise RuntimeError("Matcher could not be initialized for evaluation.")
    LOGGER.info("Matcher initialized successfully")

    allowed_track_ids_by_sequence = resolve_split_track_filters(
        dataset_root,
        sequences,
        args.train_sequences,
        args.val_sequences,
        args.val_ratio,
        args.split_seed,
        split_subset,
    )

    sequence_runs = []
    all_pairwise_decisions = []
    default_clustering_metrics_per_sequence = []

    # First pass: collect tracks and similarity scores once per sequence.
    for seq in sequences:
        default_metrics, pairwise_decisions, local_cars_registry, assignments, gt_by_camera, video_paths, camera_names, allowed_track_ids = run_matcher_on_gt(
            dataset_root,
            seq,
            allowed_track_ids=allowed_track_ids_by_sequence[seq],
        )
        sequence_runs.append(
            {
                "seq": seq,
                "default_metrics": default_metrics,
                "pairwise_decisions": pairwise_decisions,
                "local_cars_registry": local_cars_registry,
                "assignments": assignments,
                "gt_by_camera": gt_by_camera,
                "video_paths": video_paths,
                "camera_names": camera_names,
                "allowed_track_ids": allowed_track_ids,
            }
        )
        all_pairwise_decisions.extend(pairwise_decisions)
        default_clustering_metrics_per_sequence.append(
            default_metrics["final_clustering_metrics"]
        )

    os.makedirs(output_dir, exist_ok=True)
    metrics_by_threshold = []
    best_metrics = None
    best_assignments_by_sequence = {
        run["seq"]: run["assignments"] for run in sequence_runs
    }

    default_metrics_summary = {
        "pairwise_positive_prevalence": compute_pair_prevalence(all_pairwise_decisions),
        "random_baselines": {
            "fair_coin": compute_random_baseline_metrics(all_pairwise_decisions, 0.5),
            "prevalence_matched": compute_random_baseline_metrics(
                all_pairwise_decisions,
                compute_pair_prevalence(all_pairwise_decisions),
            ),
        },
        "pairwise_decision_metrics": compute_pairwise_decision_metrics(all_pairwise_decisions),
        "final_clustering_metrics": aggregate_clustering_metrics(
            default_clustering_metrics_per_sequence
        ),
        "per_sequence": {
            run["seq"]: run["default_metrics"] for run in sequence_runs
        },
    }

    for threshold in thresholds:
        pairwise_metrics = compute_pairwise_decision_metrics_at_threshold(
            all_pairwise_decisions, threshold
        )
        per_sequence_clustering_metrics = []
        threshold_assignments_by_sequence = {}
        for run in sequence_runs:
            threshold_track_manager = build_track_manager_for_threshold(
                run["local_cars_registry"],
                run["pairwise_decisions"],
                threshold,
            )
            per_sequence_clustering_metrics.append(
                compute_pairwise_metrics(threshold_track_manager)
            )
            threshold_assignments_by_sequence[run["seq"]] = (
                threshold_track_manager.local_to_global.copy()
            )
        clustering_metrics = aggregate_clustering_metrics(
            per_sequence_clustering_metrics
        )
        threshold_metrics = {
            "threshold": threshold,
            "pairwise_positive_prevalence": compute_pair_prevalence(all_pairwise_decisions),
            "random_baselines": {
                "fair_coin": compute_random_baseline_metrics(all_pairwise_decisions, 0.5),
                "prevalence_matched": compute_random_baseline_metrics(
                    all_pairwise_decisions,
                    compute_pair_prevalence(all_pairwise_decisions),
                ),
            },
            "pairwise_decision_metrics": pairwise_metrics,
            "final_clustering_metrics": clustering_metrics,
            "beats_fair_coin_f1": pairwise_metrics["f1"] > compute_random_baseline_metrics(all_pairwise_decisions, 0.5)["f1"],
            "beats_prevalence_matched_f1": pairwise_metrics["f1"] > compute_random_baseline_metrics(
                all_pairwise_decisions,
                compute_pair_prevalence(all_pairwise_decisions),
            )["f1"],
        }
        metrics_by_threshold.append(threshold_metrics)

        if best_metrics is None or pairwise_metrics["f1"] > best_metrics["pairwise_decision_metrics"]["f1"]:
            best_metrics = threshold_metrics
            best_assignments_by_sequence = threshold_assignments_by_sequence

    summary_metrics = {
        "default_matcher_threshold_metrics": default_metrics_summary,
        "threshold_sweep": metrics_by_threshold,
        "best_threshold_by_pairwise_f1": best_metrics,
    }

    render_threshold = args.render_threshold if args.render_threshold is not None else best_metrics[
        "threshold"]
    LOGGER.info("Rendering videos with threshold %.3f", render_threshold)
    serializable_assignments = {}
    for run in sequence_runs:
        render_track_manager = build_track_manager_for_threshold(
            run["local_cars_registry"],
            run["pairwise_decisions"],
            render_threshold,
        )
        render_assignment_videos(
            output_dir=output_dir,
            seq=run["seq"],
            video_paths=run["video_paths"],
            gt_by_camera=run["gt_by_camera"],
            camera_names=run["camera_names"],
            assignments=render_track_manager.local_to_global,
            allowed_track_ids=run["allowed_track_ids"],
            logger=LOGGER,
        )
        serializable_assignments[run["seq"]] = {
            f"{cam_idx}:{local_id}": global_id
            for (cam_idx, local_id), global_id in render_track_manager.local_to_global.items()
        }
    metrics_path = Path(output_dir) / f"{seq_label}_metrics.json"
    assignments_path = Path(output_dir) / f"{seq_label}_assignments.json"
    pairwise_decisions_path = Path(
        output_dir) / f"{seq_label}_pairwise_decisions.json"
    pr_curve_path = Path(output_dir) / \
        f"{seq_label}_precision_recall_curve.png"

    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    LOGGER.info("Saved metrics to %s", metrics_path)

    with open(assignments_path, "w") as f:
        json.dump(serializable_assignments, f, indent=2)
    LOGGER.info("Saved assignments to %s", assignments_path)

    with open(pairwise_decisions_path, "w") as f:
        json.dump(all_pairwise_decisions, f, indent=2)
    LOGGER.info("Saved pairwise decisions to %s", pairwise_decisions_path)

    plot_title = f"Matcher Pairwise Precision-Recall Curve {seq_label}_{split_subset}"

    plot_precision_recall_curve(
        metrics_by_threshold, all_pairwise_decisions, pr_curve_path, plot_title)

    print(f"Saved annotated videos and metrics to {output_dir}")
    print(json.dumps(summary_metrics, indent=2))
