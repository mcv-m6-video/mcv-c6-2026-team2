import numpy as np
import cv2
import torch

from scipy.optimize import linear_sum_assignment as linear_assignment

from src.models import BaseModel
from src.utils.tracker import compute_iou, filter_duplicates


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


class Track:
    """
    This class represents the internel state of individual
    tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, predominant_of_method='median'):
        """
        Initialises a tracker using initial bounding box.
        """
        self.id = Track.count
        Track.count += 1
        self.predominant_of_method = predominant_of_method
        self.bboxes = [bbox]
        self.misses = 0  # time since last update
        self.age = 0
        self.hit_streak = 0

    def last_bbox(self):
        """Returns the most recent bounding box."""
        return self.bboxes[-1]

    def update(self, bbox):
        """Updates the track with observed bbox."""
        self.bboxes.append(bbox)
        self.misses = 0
        self.hit_streak += 1

    def predict(self, of_output: np.ndarray):
        """Predict the next bounding box position using optical flow."""

        last_bbox = self.last_bbox()

        of_bbox = of_output[int(self.last_bbox()[1]):int(np.ceil(self.last_bbox()[3])),
                            int(self.last_bbox()[0]):int(np.ceil(self.last_bbox()[2]))]

        if self.predominant_of_method == 'median':
            of_vector = np.median(of_bbox.reshape(-1, 2), axis=0)
        elif self.predominant_of_method == 'mean':
            of_vector = np.mean(of_bbox.reshape(-1, 2), axis=0)

        predicted_bbox = [
            int(last_bbox[0] + of_vector[0]),
            int(last_bbox[1] + of_vector[1]),
            int(last_bbox[2] + of_vector[0]),
            int(last_bbox[3] + of_vector[1]),
        ]

        self.age += 1
        if self.misses > 0:
            self.hit_streak = 0

        return predicted_bbox


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)),
                np.empty((0, 5), dtype=int))
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    row_ind, col_ind = linear_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class OFTracker:
    """Object tracker that combines object detection and optical flow for tracking across frames.

    Args:
    iou_threshold: IOU threshold for matching predicted tracks with detections;
    dup_iou_threshold: IOU threshold for filtering duplicate detections;
    max_age: maximum number of frames to keep a track without updates before deletion;
    min_hits: minimum number of hits to consider a track as valid;
    conf_threshold: confidence threshold for filtering detections.
    """

    def __init__(
        self,
        of: BaseModel,
        obj_detector,
        iou_threshold: float = 0.4,
        dup_iou_threshold: float = 0.9,
        max_age: int = 5,
        min_hits: int = 3,
        conf_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.of = of
        self.obj_detector = obj_detector
        self.iou_threshold = iou_threshold
        self.dup_iou_threshold = dup_iou_threshold
        self.max_age = max_age
        self.conf_threshold = conf_threshold

        self.tracks: list[Track] = []
        self.frame_count = 0

        self.max_age = max_age
        self.min_hits = min_hits

        self.device = device

    def initialize_tracks(self, initial_frame: np.ndarray):
        """Initialize tracks with initial detections."""
        initial_dets = self.detect_and_filter(initial_frame)[0]
        for det in initial_dets:
            self.tracks.append(Track(det))

    def detect_all_frames(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[np.ndarray]:
        """
        Run object detection in batches for all frames.

        Args:
            frames: list of video frames
            batch_size: number of frames per detection batch

        Returns:
            all_dets: list where each element has shape [N_i, 4]
        """
        all_dets = []

        for start in range(0, len(frames), batch_size):
            end = min(start + batch_size, len(frames))
            print(f"[Offline] Detecting frames {start}..{end-1}", flush=True)

            batch_frames = frames[start:end]
            batch_dets = self.detect_and_filter(
                batch_frames)  # already returns list

            all_dets.extend(batch_dets)

        return all_dets

    def detect_and_filter(self, image: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
        """Detect objects in one or more images and filter detections.

        Args:
            image: single image or list of images as numpy arrays

        Returns:
            dets_parsed: list of arrays, each of shape [N_i, 4]
        """
        if isinstance(image, np.ndarray):
            image = [image]

        elif not (isinstance(image, list) and all(isinstance(img, np.ndarray) for img in image)):
            raise TypeError(
                "image must be a numpy array or a list of numpy arrays")

        img_list = []
        for img in image:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(
                img_rgb).permute(2, 0, 1).float() / 255.0
            img_list.append(img_tensor.to(device=self.device))

        with torch.no_grad():
            dets = self.obj_detector(img_list)

        dets_parsed = []
        for frame in dets:
            dets_frame = []
            for box_i, box in enumerate(frame["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                conf = frame["scores"][box_i].item()

                if conf >= self.conf_threshold:
                    dets_frame.append([x1, y1, x2, y2, conf])

            dets_frame = filter_duplicates(
                dets_frame, threshold=self.dup_iou_threshold)

            if len(dets_frame) == 0:
                dets_parsed.append(np.empty((0, 4), dtype=float))
            else:
                dets_parsed.append(np.array(dets_frame, dtype=float)[:, :4])

        return dets_parsed

    def update(self, image0: np.ndarray, image1: np.ndarray):
        """Update tracks with new detections and optical flow predictions.

        Args:
            image0: previous frame as a numpy array
            image1: current frame as a numpy array
        """
        self.frame_count += 1

        # placeholder for predicted tracks; shape Nx5: [x1,y1,x2,y2,track_id]
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []

        # get optical flow predictions
        of_output = self.of([image0, image1])

        for t, track in enumerate(trks):
            # Update track with predicted bbox from optical flow
            predicted_bbox = self.tracks[t].predict(of_output)
            track[:] = [predicted_bbox[0], predicted_bbox[1],
                        predicted_bbox[2], predicted_bbox[3], 0]

            if np.any(np.isnan(predicted_bbox)):
                to_del.append(t)

        # Remove tracks with invalid predictions (nans of infs)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        # Get detections for current frame
        dets1 = self.detect_and_filter(image1)[0]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets1, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.tracks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets1[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = Track(dets1[i, :])
            self.tracks.append(trk)
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.last_bbox()
            if ((trk.misses < 1) and
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.misses > self.max_age):
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def compute_all_flows(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute optical flow for all consecutive frame pairs.
        flows[i] corresponds to flow from frames[i] -> frames[i+1].
        """
        all_flows = []

        for i in range(len(frames) - 1):
            print(f"[Offline] Optical flow {i}->{i+1}", flush=True)
            flow = self.of([frames[i], frames[i + 1]])
            all_flows.append(flow)

        return all_flows

    def initialize_tracks_from_dets(self, initial_dets: np.ndarray):
        """Initialize tracks from already computed detections."""
        self.tracks = []
        self.frame_count = 0
        for det in initial_dets:
            self.tracks.append(Track(det))

    def update_from_precomputed(
        self,
        of_output: np.ndarray,
        dets1: np.ndarray,
    ):
        """
        Update tracks using precomputed optical flow and detections.
        dets1 must be shape [N, 4].
        """
        self.frame_count += 1

        trks = np.zeros((len(self.tracks), 5), dtype=float)
        to_del = []
        ret = []

        # Predict existing tracks with precomputed flow
        for t, track_row in enumerate(trks):
            predicted_bbox = self.tracks[t].predict(of_output)
            track_row[:] = [
                predicted_bbox[0],
                predicted_bbox[1],
                predicted_bbox[2],
                predicted_bbox[3],
                0,
            ]

            if np.any(np.isnan(predicted_bbox)):
                to_del.append(t)

        # Remove invalid predictions
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        # Associate detections with predicted tracks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets1, trks, self.iou_threshold
        )

        # Update matched trackers
        for t, trk in enumerate(self.tracks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets1[d, :][0])
            else:
                # unmatched track -> keep predicted position and count miss
                trk.bboxes.append(trks[t, :4].tolist())
                trk.misses += 1
                trk.age += 1
                trk.hit_streak = 0

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            self.tracks.append(Track(dets1[i, :]))

        # Prepare output and remove dead tracks
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.last_bbox()
            if (
                (trk.misses < 1)
                and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

            i -= 1
            if trk.misses > self.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5))
