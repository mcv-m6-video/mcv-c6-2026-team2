"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

#Constants
INFERENCE_BATCH_SIZE = 4
INFERENCE_NUM_WORKERS = 2

def evaluate(model, dataset, batch_size=INFERENCE_BATCH_SIZE, num_workers=INFERENCE_NUM_WORKERS):
    # Initialize scores and labels
    scores = []
    labels = []
    # Perform inference
    for clip in tqdm(DataLoader(
            dataset, num_workers=num_workers, pin_memory=True,
            batch_size=batch_size
    )):
        # Batched by dataloader
        batch_pred_scores = model.predict(clip['frame'])
        label = clip['label'].cpu().numpy()
        # Store scores and labels
        scores.append(batch_pred_scores)
        labels.append(label)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    ap_score = average_precision_score(labels, scores, average=None)  # Set to None so AP per class are not averaged

    return ap_score