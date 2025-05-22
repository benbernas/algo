# preprocess_mb
"""
Created on Mon May 19 19:34:18 2025

@author: Benoît  Bernas
"""

"""
Pre-processing of Train / Test assemblies :
    1.  Remove samples where two micro-bubbles (MB) are closer
        than the threshold (in pixels).
    2.  Then converts MB coordinates (mm) → pixels.
    3.  Recovers the corresponding intensities and reorders
        each image / set of coordinates by decreasing intensity.

Fonctions
------------------
clean_and_rank(train_IQ, train_coord_mm,
               test_IQ,  test_coord_mm,
               zExtent, xExtent, dz, dx,
               dist_threshold_px=0.105)

Outputs :
    (IQ_train, coord_train_desc_mm,
     IQ_test,  coord_test_desc_mm,
     infos_train, infos_test)

Each “infos_...” contains: intensities, sorted indices, sorted intensities..
"""

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------
#   1.  -- suppression des échantillons trop proches ------------------
# ---------------------------------------------------------------------
def _filter_min_dist(coord_mm, IQ, threshold_px):
    """
    Removes images where the minimum distance between two MBs is
    strictly less than `threshold_px`.

    Parameters
    ----------
    coord_mm : (N, 2, nMB)
    IQ : (N, depth, width)
    threshold_px : float (threshold expressed in pixels)

    Returns
    -------
    coord_mm_clean, IQ_clean
    """
    min_dists = []
    for n in range(coord_mm.shape[0]):
        # Euclidean distance matrix  (nMB × nMB)
        dist_mat = cdist(coord_mm[n].T, coord_mm[n].T)
        np.fill_diagonal(dist_mat, np.inf)
        min_dists.append(np.min(dist_mat))

    bad_idx = np.where(np.array(min_dists) < threshold_px)[0]
    coord_clean = np.delete(coord_mm, bad_idx, axis=0)
    IQ_clean    = np.delete(IQ,       bad_idx, axis=0)
    return coord_clean, IQ_clean


# ---------------------------------------------------------------------
#   2.  -- converting mm → pixel --------------------------------------
# ---------------------------------------------------------------------
def _mm_to_px(coord_mm, zExtent, xExtent, dz, dx):
    """
    Converts coordinates (mm) to pixels.

    Returns
    -------
    x_px, z_px  (int, shape = coord_mm[...,0].shape)
    """
    x_mm, z_mm = coord_mm[:, 0, :], coord_mm[:, 1, :]
    x_px = np.round((x_mm - xExtent[0]) / dx).astype(int)
    z_px = np.round((z_mm - zExtent[0]) / dz).astype(int)
    return x_px, z_px


# ---------------------------------------------------------------------
#   3.  -- ranking by intensity ---------------------------------------
# ---------------------------------------------------------------------
def _rank_by_intensity(IQ, x_px, z_px, coord_mm):
    """
    For each image :
        - retrieves the MB intensities ;
        - sorts indices by decreasing intensity ;
        - reorders coord_mm accordingly (flatten).

    Returns three tables:
        intensities, sorted_indices, intensities_desc
        coord_mm_desc (N, 2*nMB)
    """
    intensities, indices_sorted, intensities_desc, coord_desc = [], [], [], []
    for i in range(coord_mm.shape[0]):
        intens = IQ[i, z_px[i], x_px[i]]               # nMB
        idx_desc = np.flip(np.argsort(intens))         # decreasing intensity

        intensities.append(intens)
        indices_sorted.append(idx_desc)
        intensities_desc.append(intens[idx_desc])

        coord_desc.append(coord_mm[i][:, idx_desc].flatten())

    return (np.array(intensities),
            np.array(indices_sorted),
            np.array(intensities_desc),
            np.array(coord_desc))


# ---------------------------------------------------------------------
#   4.  -- main function  ---------------------------------------------
# ---------------------------------------------------------------------
def clean_and_rank(train_IQ, train_coord_mm,
                   test_IQ,  test_coord_mm,
                   zExtent, xExtent, dz, dx,
                   dist_threshold_px=0.105):
    """
    Performs all the steps described in the file header.
    """

    # 1) ─── neighbourhood MBs filter ─────────────────────────────────────────
    coord_train, IQ_train = _filter_min_dist(train_coord_mm,
                                             train_IQ,
                                             dist_threshold_px)
    coord_test,  IQ_test  = _filter_min_dist(test_coord_mm,
                                             test_IQ,
                                             dist_threshold_px)

    # 2) ─── converting mm → px ───────────────────────────────────────
    x_px_tr, z_px_tr = _mm_to_px(coord_train, zExtent, xExtent, dz, dx)
    x_px_te, z_px_te = _mm_to_px(coord_test,  zExtent, xExtent, dz, dx)

    # 3) ─── ranking by intensity ────────────────────────────────────
    infos_tr = _rank_by_intensity(IQ_train, x_px_tr, z_px_tr, coord_train)
    infos_te = _rank_by_intensity(IQ_test,  x_px_te, z_px_te, coord_test)

    # re-ordered coordinates (mm) for learning
    coord_train_desc_mm = infos_tr[-1]
    coord_test_desc_mm  = infos_te[-1]

    return (IQ_train, coord_train_desc_mm,
            IQ_test,  coord_test_desc_mm,
            infos_tr, infos_te)
