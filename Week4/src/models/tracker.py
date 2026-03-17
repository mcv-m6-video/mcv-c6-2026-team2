from src.utils.tracker import compute_iou

class Track:
    def __init__(self, initial_bbox, track_id):
        self.id = track_id
        self.bboxes = [initial_bbox]  # [xtl, ytl, xbr, ybr]
        self.misses = 0

    def last_bbox(self):
        return self.bboxes[-1]

    def update(self, new_bbox):
        self.bboxes.append(new_bbox)
        self.misses = 0

class OverlapTracker:
    def __init__(self, iou_threshold=0.4, max_age=5, conf_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.conf_threshold = conf_threshold
        self.active_tracks = []
        self.next_id = 0

    def update(self, new_detections):
        # Filter per confidence
        detections = [d for d in new_detections if d[4] >= self.conf_threshold]

        matched_indices = set()
        updated_tracks = []

        # Match existent tracks
        for track in self.active_tracks:
            best_iou = 0
            best_det_idx = None

            for i, det in enumerate(detections):
                if i in matched_indices:
                    continue

                iou = compute_iou(track.last_bbox(), det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx is not None and best_iou >= self.iou_threshold:
                track.update(detections[best_det_idx][:4])
                matched_indices.add(best_det_idx)
                updated_tracks.append(track)
            else:
                track.misses += 1
                if track.misses <= self.max_age:
                    updated_tracks.append(track)

        # Create new tracks
        for i, det in enumerate(detections):
            if i not in matched_indices:
                new_track = Track(det[:4], self.next_id)
                updated_tracks.append(new_track)
                self.next_id += 1

        self.active_tracks = updated_tracks
        return self.active_tracks