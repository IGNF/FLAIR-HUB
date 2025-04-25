import numpy as np

def apply_numpy_augmentations(batch_dict, input_keys, label_keys, p_flip=0.5, p_rot=0.5):
    """
    Applies the same random 2D augmentations to all input and label arrays within a single sample.
    Useful when __getitem__ returns a single sample with potentially multiple modalities.

    Args:
        batch_dict (dict): A single sample dict with keys for inputs and labels.
        input_keys (list): Keys of input arrays to augment.
        label_keys (list): Keys of label arrays to augment.
        p_flip (float): Probability of horizontal/vertical flip.
        p_rot (float): Probability of 90° rotation.

    Returns:
        dict: Augmented batch_dict.
    """

    def apply_transforms(arr):
        if do_hflip:
            arr = np.flip(arr, axis=-1)
        if do_vflip:
            arr = np.flip(arr, axis=-2)
        if k_rot > 0:
            arr = np.rot90(arr, k=k_rot, axes=(-2, -1))
        return arr

    # Generate one set of random decisions
    do_hflip = np.random.rand() < p_flip
    do_vflip = np.random.rand() < p_flip
    do_rot = np.random.rand() < p_rot
    k_rot = np.random.randint(1, 4) if do_rot else 0

    # Apply the same transformation to each input and label key
    for key in input_keys + label_keys:
        arr = batch_dict[key]
        shape = arr.shape

        # Flatten leading dimensions so we apply transforms to the last two (H, W)
        reshaped = arr.reshape(-1, *shape[-2:])
        for i in range(reshaped.shape[0]):
            reshaped[i] = apply_transforms(reshaped[i])
        batch_dict[key] = reshaped.reshape(shape)

    return batch_dict
