import os
from src.task2.overlap_tracker import OverlapTracker
from src.task2.utils import load_maskrcnn_detections, filter_duplicates, convert_xml_to_mot, run_trackeval_script, create_tracking_video, prepare_trackeval_folders

def run_task21(det_path, output_txt_path, video_path=None, xml_gt_path=None, trackeval_path=None, make_video=False, iou_threshold=0.4):
    """
    Main execution for Task 2.1: Tracking by Maximum Overlap.
    """
    all_detections = load_maskrcnn_detections(det_path)
    
    tracker = OverlapTracker(iou_threshold=iou_threshold)
    all_results = []

    for frame_id in sorted(all_detections.keys()):
        raw_dets = all_detections[frame_id]
        clean_dets = filter_duplicates(raw_dets, threshold=0.9)
        active_tracks = tracker.update(clean_dets)
        
        for track in active_tracks:
            if track.active:
                bbox = track.last_bbox() # [xtl, ytl, xbr, ybr]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                # Format: <frame>, <id>, <left>, <top>, <width>, <height>, <conf>, -1, -1, -1
                all_results.append(f"{frame_id},{track.id},{bbox[0]:.2f},{bbox[1]:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w') as f:
        f.writelines(all_results)
    print(f"Tracking results saved to: {output_txt_path}")

    if xml_gt_path and trackeval_path:
        gt_txt_path = output_txt_path.replace(".txt", "_gt.txt")
        convert_xml_to_mot(xml_gt_path, gt_txt_path)
        print(f"Ground Truth converted for TrackEval: {gt_txt_path}")
        print("Starting Evaluation with TrackEval...")
        prepare_trackeval_folders(trackeval_path, output_txt_path, gt_txt_path, tracker_name="overlap")
        run_trackeval_script(trackeval_path, tracker_name="overlap")

    if video_path and make_video:
        video_out = output_txt_path.replace(".txt", "_viz.mp4")
        print(f"Generating video: {video_out}...")
        create_tracking_video(video_path, output_txt_path, video_out)
        print("Video generation completed.")

    return output_txt_path