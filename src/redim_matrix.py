""" 
Redimension a matrix :
    1) line = object (use mean or median spectrum of the object)
    2) line = pixel (use all pixels of the object)
    3) line = m pixels (use m random pixels of the object)
"""

import numpy as np

def object_db_to_balanced_px_matrix(
        object_db, 
        m=40, 
        allowed_splits=None, 
        allowed_labels=None,
        random_state=42,
        replace=False,
    ):
    """
    Convert object database to a balanced pixel matrix.
    Each object contributes the same number of pixels m whenever possible

    Parameters
    ----------
    object_db : dict
        Object-level database.
    m : int
        Number of pixels sampled per object.
    allowed_splits : list or None
        Keep only objects whose obj["split"] is in this list.
        Example: ["train_minimal"]
    allowed_labels : list or None
        Keep only objects whose obj["object_nut_type"] is in this list.
        Example: ["almond", "peanut"]
    random_state : int
        Seed for reproducible random sampling.
    replace : bool
        If False:
            an object with fewer than m pixels contributes all its pixels.
        If True:
            an object with fewer than m pixels is sampled with replacement
            so it still contributes exactly m rows.

    Returns
    -------
    X : ndarray, shape (N, B)
        Balanced pixel matrix.
    y : ndarray, shape (N,)
        Class label for each selected pixel.
    pixel_object_ids : ndarray, shape (N,)
        Object ID for each selected pixel.
    pixel_source_images : ndarray, shape (N,)
        Source image for each selected pixel.
    pixel_positions : ndarray, shape (N, 2)
        Global image coordinates of each selected pixel.
    """
    rng = np.random.default_rng(random_state)
    X_list = []                 # selected spectra
    y_list = []                 # pixels labels
    pixel_object_ids = []       # original object
    pixel_source_images = []    # original source image
    pixel_positions = []        # original pixel position within the object

    for obj_id, obj in object_db.items():
        if allowed_splits is not None and obj["split"] not in allowed_splits:
            continue
        if allowed_labels is not None and obj["object_nut_type"] not in allowed_labels:
            continue   

        spectra = obj["spectra"]  # shape (n_pixels, n_bands)
        positions = obj["positions_global"] # (n_pixels, 2)
        n_k = spectra.shape[0] # number of pixels in the object

        # randomly select m pixels from the object
        if n_k == 0:
            continue
        if n_k >=m:
            idx = rng.choice(n_k, size=m, replace=False)
        else:
            if replace:
                idx = rng.choice(n_k, size=m, replace=True) # if not enough pixels, replacement with repetition of existing pixels
            else:
                idx = np.arange(n_k)
        X_selected = spectra[idx]
        pos_selected = positions[idx]
        X_list.append(X_selected)
        n_selected = X_selected.shape[0]
        y_list.extend([obj["object_nut_type"]] * n_selected)    # each pixel inherits the label & etc of the object
        pixel_object_ids.extend([obj_id] * n_selected)
        pixel_source_images.extend([obj["source_clean_key"]] * n_selected)
        pixel_positions.append(pos_selected)    # (row, col) position of the pixel within the original image

    X = np.vstack(X_list)
    y = np.array(y_list)
    pixel_object_ids = np.array(pixel_object_ids)
    pixel_source_images = np.array(pixel_source_images)
    pixel_positions = np.vstack(pixel_positions)

    return X, y, pixel_object_ids, pixel_source_images, pixel_positions



def object_db_to_object_matrix(
    object_db,
    spectrum_field="mean_spectrum",
    allowed_splits=None,
    allowed_labels=None,
):
    """ Convert object database to a matrix where each line is an object spectrum.

    Parameters
    ----------
    object_db : dict
        Dictionary of objects.
    spectrum_field : str, optional
        Field name for the spectrum data. Defaults to "mean_spectrum".
        Can be "mean_spectrum" or "median_spectrum" (1 value per object)
    allowed_splits : list or None, optional
        List of allowed splits. Defaults to None.
    allowed_labels : list or None, optional
        List of allowed labels. Defaults to None.

    Returns
    -------
    X : ndarray, shape (N, B)
        Object spectrum matrix.
    y : ndarray, shape (N,)
        Class label for each object.
    object_ids : ndarray, shape (N,)
        Object ID for each object.
    source_images : ndarray, shape (N,)
        Source image for each object.
    batches : ndarray, shape (N,)
        Batch for each object.
    areas : ndarray, shape (N,)
        Area of each object.
    """
    X_list = []
    y_list = []
    object_ids = []
    source_images = []
    batches = []
    areas = []

    for obj_id, obj in object_db.items():
        if allowed_splits is not None and obj["split"] not in allowed_splits:
            continue
        if allowed_labels is not None and obj["object_nut_type"] not in allowed_labels:
            continue

        X_list.append(obj[spectrum_field])
        y_list.append(obj["object_nut_type"])
        object_ids.append(obj_id)
        source_images.append(obj["source_clean_key"])
        batches.append(obj["batch"])
        areas.append(obj["area_pixels"])

    X = np.vstack(X_list)
    y = np.array(y_list)
    object_ids = np.array(object_ids)
    source_images = np.array(source_images)
    batches = np.array(batches)
    areas = np.array(areas)

    return X, y, object_ids, source_images, batches, areas



def object_db_to_pixel_matrix(
    object_db,
    allowed_splits=None,
    allowed_labels=None,
):
    X_list = []
    y_list = []
    object_ids = []
    source_images = []
    positions = []

    for obj_id, obj in object_db.items():
        if allowed_splits is not None and obj["split"] not in allowed_splits:
            continue
        if allowed_labels is not None and obj["object_nut_type"] not in allowed_labels:
            continue

        spectra = obj["spectra"]
        n_pixels = spectra.shape[0]
        X_list.append(spectra)
        y_list.extend([obj["object_nut_type"]] * n_pixels)
        object_ids.extend([obj_id] * n_pixels)
        source_images.extend([obj["source_clean_key"]] * n_pixels)
        positions.append(obj["positions_global"])

    X = np.vstack(X_list)
    y = np.array(y_list)
    object_ids = np.array(object_ids)
    source_images = np.array(source_images)
    positions = np.vstack(positions)

    return X, y, object_ids, source_images, positions