import pandas as pd
import argparse


def split_gt_by_sequence(input_file, sequence_map, output_dir="."):
    """
    Divide the ground_truth_train.txt in sveral txts (one per sequence).

    Args:
        input_file (str): ground_truth_trin.txt path
        sequence_map (dict): { "S01": [cam_ids], ... }
        output_dir (str): output path
    """

    cols = ['CameraId','Id','FrameId','X','Y','Width','Height','Xworld','Yworld']

    df = pd.read_csv(input_file, header=None, names=cols, sep=r'\s+|,')

    results = {}

    for seq_name, cam_ids in sequence_map.items():
        df_seq = df[df['CameraId'].isin(cam_ids)]

        out_file = f"{output_dir}/gt_{seq_name}.txt"
        df_seq.to_csv(out_file, header=False, index=False)

        print(f"{seq_name}: {len(df_seq)} rows → {out_file}")
        results[seq_name] = out_file

    return results


def main():
    parser = argparse.ArgumentParser(description="Split GT by sequence")
    parser.add_argument("--input", required=True, help="Path to ground_truth_train.txt")
    parser.add_argument("--output_dir", default=".", help="Output directory")

    args = parser.parse_args()

    sequence_map = {
        "S01": [1,2,3,4,5],
        "S03": [10,11,12,13,14,15],
        "S04": [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
    }

    split_gt_by_sequence(args.input, sequence_map, args.output_dir)

if __name__ == "__main__":
    main()