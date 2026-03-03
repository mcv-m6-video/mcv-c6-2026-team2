def filter_predictions(
    prediction: list[dict], target_class: int, id2label: dict, threshold: float
):
    filtered_predictions = []

    for pred in prediction:
        mask = (pred["scores"] > threshold) & (pred["labels"] == target_class)
        filtered_predictions.append(
            {
                "boxes": pred["boxes"][mask].cpu(),
                "labels": pred["labels"][mask].cpu(),
                "scores": pred["scores"][mask].cpu(),
            }
        )

    return filtered_predictions


def save_prediction_txt(
    prediction: list[dict], output_file: str, frame_id: list[int], threshold: float = 0.5
):
    boxes = prediction["boxes"].detach().numpy()
    scores = prediction["scores"].detach().numpy()

    with open(output_file, "a") as f:
        for i in range(len(boxes)):
            if scores[i] > threshold:
                xtl, ytl, xbr, ybr = boxes[i]

                w = xbr - xtl
                h = ybr - ytl

                line = f"{frame_id},-1,{xtl:.3f},{ytl:.3f},{w:.3f},{h:.3f},{scores[i]:.3f}\n"
                f.write(line)
