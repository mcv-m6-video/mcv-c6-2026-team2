import os
import shutil
import subprocess

def prepare_trackeval_folders(
    base_trackeval_path, results_path, gt_path, tracker_name="overlap"
):
    """
    Creates the exact MOTChallenge directory structure expected by TrackEval.
    """
    # Define the benchmark name and sequence name consistently
    benchmark = "AICity"
    split = "train"
    sequence = "s03"  # This is the folder name TrackEval will look for

    # Define internal TrackEval paths
    # TrackEval/data/gt/mot_challenge/AICity-train/s03/gt/gt.txt
    # TrackEval/data/trackers/mot_challenge/AICity-train/overlap/data/s03.txt
    data_path = os.path.join(base_trackeval_path, "data")
    gt_base_path = os.path.join(
        data_path, "gt", "mot_challenge", f"{benchmark}-{split}"
    )
    seqmap_path = os.path.join(data_path, "gt", "mot_challenge", "seqmaps")
    tracker_data_path = os.path.join(
        data_path,
        "trackers",
        "mot_challenge",
        f"{benchmark}-{split}",
        tracker_name,
        "data",
    )

    # Create the directory tree
    os.makedirs(os.path.join(gt_base_path, sequence, "gt"), exist_ok=True)
    os.makedirs(seqmap_path, exist_ok=True)
    os.makedirs(tracker_data_path, exist_ok=True)

    seq_length = 0
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                frame_idx = int(line.split(',')[0])
                if frame_idx > seq_length:
                    seq_length = frame_idx

    # Create seqinfo.ini (Crucial for TrackEval to know the sequence length)
    ini_content = f"[Sequence]\nname={sequence}\nseqLength={seq_length}\n"
    with open(os.path.join(gt_base_path, sequence, "seqinfo.ini"), "w") as f:
        f.write(ini_content)

    # Create the seqmap file (Must be named: [BENCHMARK]-[SPLIT].txt)
    with open(os.path.join(seqmap_path, f"{benchmark}-{split}.txt"), "w") as f:
        f.write(f"name\n{sequence}")

    # Copy Ground Truth and Predictions
    # Prediction file MUST be named after the sequence (s03.txt)
    shutil.copy(results_path, os.path.join(tracker_data_path, f"{sequence}.txt"))
    # GT file MUST be named gt.txt
    shutil.copy(gt_path, os.path.join(gt_base_path, sequence, "gt", "gt.txt"))

    print(
        f"TrackEval structure ready for benchmark '{benchmark}' and tracker "
        f"'{tracker_name}'"
    )


def run_trackeval_script(trackeval_path, tracker_name="overlap", save_path=None):
    """
    Runs the TrackEval script using the standardized folder structure created above.
    """
    command = [
        "python", os.path.join(trackeval_path, "scripts",
                               "run_mot_challenge.py"),
        "--BENCHMARK", "AICity",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS", "HOTA", "Identity", "CLEAR",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if save_path is not None:
        print(save_path)
        with open(save_path, "w") as f:
            f.write(result.stdout)
        print(f"Metrics saved to: {save_path}")
