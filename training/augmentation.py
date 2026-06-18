"""
ChiliRover AI — Augmentation Presets
=====================================
Returns augmentation keyword arguments tuned for detecting
early-stage chili leaf disease spots from a moving rover camera.
"""


def get_augmentation_kwargs(config: dict) -> dict:
    """
    Build the augmentation keyword dict consumed by YOLO .train().

    Parameters
    ----------
    config : dict
        The full pipeline config (parsed from pipeline_config.yaml).

    Returns
    -------
    dict
        Augmentation kwargs ready to unpack into model.train(**kwargs).
    """
    aug = config.get("augmentation", {})

    return {
        # Colour-space — high saturation jitter is critical for
        # distinguishing subtle early-stage spots from healthy tissue.
        "hsv_h": aug.get("hsv_h", 0.02),
        "hsv_s": aug.get("hsv_s", 0.70),
        "hsv_v": aug.get("hsv_v", 0.40),

        # Spatial — simulate the rover's variable viewing angles.
        "degrees": aug.get("degrees", 15.0),
        "translate": aug.get("translate", 0.15),
        "scale": aug.get("scale", 0.50),
        "shear": aug.get("shear", 5.0),

        # Flips
        "flipud": aug.get("flipud", 0.3),
        "fliplr": aug.get("fliplr", 0.5),

        # Composition — more objects per training image.
        "mosaic": aug.get("mosaic", 1.0),
        "mixup": aug.get("mixup", 0.15),
    }
