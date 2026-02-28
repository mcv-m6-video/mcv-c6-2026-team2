from src.task2.utils import compute_iou

class Track:
    def __init__(self, initial_bbox, track_id):
        self.id = track_id
        self.bboxes = [initial_bbox] # [x, y, w, h]
        self.active = True

    def last_bbox(self):
        return self.bboxes[-1]

    def update(self, new_bbox):
        self.bboxes.append(new_bbox)

class OverlapTracker:
    def __init__(self, iou_threshold=0.4):
        self.iou_threshold = iou_threshold
        self.active_tracks = []
        self.next_id = 0

    def update(self, new_detections):
        """
        Main logic for Task 2.1: Maximum Overlap Association.
        """
        matched_indices = set()
        updated_active_tracks = []

        # Compare active tracks from frame N to detections in frame N+1 
        for track in self.active_tracks:
            best_iou = -1
            best_det_idx = -1
            
            for i, det in enumerate(new_detections):
                if i in matched_indices: continue
                
                # Bbox format expected: [xtl, ytl, xbr, ybr]
                current_iou = compute_iou(track.last_bbox(), det[:4])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_det_idx = i

            # If max IoU >= threshold, assign existing track ID 
            if best_iou >= self.iou_threshold:
                track.update(new_detections[best_det_idx][:4])
                matched_indices.add(best_det_idx)
                updated_active_tracks.append(track)
            else:
                # If no match is found, the track becomes inactive
                track.active = False

        # Create new tracks for unmatched detections 
        for i, det in enumerate(new_detections):
            if i not in matched_indices:
                new_track = Track(det[:4], self.next_id)
                updated_active_tracks.append(new_track)
                self.next_id += 1
                
        self.active_tracks = updated_active_tracks
        return self.active_tracks