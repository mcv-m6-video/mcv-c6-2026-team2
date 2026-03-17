import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.utils.reid_training_utils import ReIDNet, build_transforms


LOGGER = logging.getLogger(__name__)
DEFAULT_SIMILARITY_THRESHOLD = 0.85
_MATCHER = None


class ReIDMatcher:
    """Loads a trained ReID model and compares cropped car images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.similarity_threshold = similarity_threshold

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        train_args = checkpoint.get("args", {})
        model_state = checkpoint["model_state_dict"]

        backbone = train_args.get("backbone", "densenet121")
        embedding_dim = train_args.get("embedding_dim", 512)
        img_size = train_args.get("img_size", 256)
        num_classes = model_state["classifier.weight"].shape[0]

        self.model = ReIDNet(
            num_classes=num_classes,
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        _, self.transform = build_transforms(img_size)

        LOGGER.info(
            "Loaded ReID matcher | checkpoint=%s | device=%s | backbone=%s | embedding_dim=%d | threshold=%.3f",
            self.checkpoint_path,
            self.device,
            backbone,
            embedding_dim,
            self.similarity_threshold,
        )

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Converts a BGR crop into a normalized input tensor."""
        if image is None or image.size == 0:
            raise ValueError("Cannot run matcher inference on an empty image crop.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def embed_image(self, image: np.ndarray) -> torch.Tensor:
        """Extracts a normalized embedding from a single crop."""
        tensor = self._preprocess(image)
        embedding, _ = self.model(tensor)
        return embedding.squeeze(0)

    @torch.no_grad()
    def average_embeddings(self, embeddings: Sequence[np.ndarray | torch.Tensor]) -> torch.Tensor:
        """Averages and normalizes a list of embeddings."""
        if len(embeddings) == 0:
            raise ValueError("Cannot compute an embedding from an empty embedding list.")

        tensor_embeddings = []
        for embedding in embeddings:
            if isinstance(embedding, np.ndarray):
                tensor_embeddings.append(torch.from_numpy(embedding).to(self.device))
            else:
                tensor_embeddings.append(embedding.to(self.device))

        mean_embedding = torch.stack(tensor_embeddings, dim=0).mean(dim=0)
        return F.normalize(mean_embedding, dim=0)

    @torch.no_grad()
    def similarity(
        self,
        embeddings_a: Sequence[np.ndarray | torch.Tensor],
        embeddings_b: Sequence[np.ndarray | torch.Tensor],
    ) -> float:
        """Computes cosine similarity between two averaged car embeddings."""
        emb_a = self.average_embeddings(embeddings_a)
        emb_b = self.average_embeddings(embeddings_b)
        similarity = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        return similarity

    def is_same_car(
        self,
        embeddings_a: Sequence[np.ndarray | torch.Tensor],
        embeddings_b: Sequence[np.ndarray | torch.Tensor],
        threshold: Optional[float] = None,
    ) -> bool:
        """Returns whether two embedding sequences likely belong to the same vehicle."""
        similarity = self.similarity(embeddings_a, embeddings_b)
        decision_threshold = self.similarity_threshold if threshold is None else threshold
        LOGGER.debug(
            "Matcher comparison | num_emb_a=%d | num_emb_b=%d | similarity=%.4f | threshold=%.4f",
            len(embeddings_a),
            len(embeddings_b),
            similarity,
            decision_threshold,
        )
        return similarity >= decision_threshold


def initialize_matcher(
    checkpoint_path: str,
    device: Optional[str] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> ReIDMatcher:
    """Initializes the global matcher instance used by Car.__eq__."""
    global _MATCHER
    _MATCHER = ReIDMatcher(
        checkpoint_path=checkpoint_path,
        device=device,
        similarity_threshold=similarity_threshold,
    )
    return _MATCHER


def get_matcher() -> Optional[ReIDMatcher]:
    """Returns the global matcher instance if it has been initialized."""
    return _MATCHER


def compare_car_embeddings(
    embeddings_a: Sequence[np.ndarray | torch.Tensor],
    embeddings_b: Sequence[np.ndarray | torch.Tensor],
    threshold: Optional[float] = None,
) -> bool:
    """Compares two cars from their stored embedding histories."""
    matcher = get_matcher()
    if matcher is None:
        LOGGER.warning(
            "ReID matcher has not been initialized yet. Returning False for car comparison."
        )
        return False
    return matcher.is_same_car(embeddings_a, embeddings_b, threshold=threshold)
