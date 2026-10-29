"""Monge-Kantorovich Linearization (MKL) colour transfer in plain numpy (MPI-1341).

Ported from the `color-matcher` package (https://github.com/hahnec/color-matcher, ColorMatcher with
method "mkl": top_level.py, mvgd_matcher.py, baseclass.py), GPL-3.0, Copyright (c) 2019-2020
Christopher Hahne. See LICENSE-color-matcher. Only the "mkl" method is carried: it is the one
Cubric Studio's workflows set. Pitie et al., "The linear Monge-Kantorovitch linear colour mapping
for example-based colour transfer" is the method; this is the same arithmetic as the package, so
the result is bit-identical to ColorMatcher().transfer(src, ref, method="mkl").

No dependency beyond numpy.
"""
import numpy as np


def _mkl_matrix(cov_src, cov_ref):
    """Transfer matrix that maps the source colour covariance onto the reference one."""
    eig_val_r, eig_vec_r = np.linalg.eig(cov_src)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1. / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_ref @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    return vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T


def mkl_transfer(src, ref):
    """Colour-match `src` (H, W, C) to `ref` (H', W', C'). C and C' are 3 or 4; alpha is dropped
    from whichever side has it when the other has none, as color-matcher does. Returns an
    (H, W, 3) float64 array, not clamped."""
    for name, img in (("source", src), ("reference", ref)):
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f"colour match needs 3 or 4 channel images, got the {name} with shape {img.shape}")
    if src.shape[2] == 3 and ref.shape[2] == 4:
        ref = ref[..., :3]
    elif src.shape[2] == 4 and ref.shape[2] == 3:
        src = src[..., :3]

    r = src.reshape([-1, src.shape[2]]).T
    z = ref.reshape([-1, ref.shape[2]]).T
    cov_r, cov_z = np.cov(r), np.cov(z)
    mu_r = r.mean(axis=1)[..., np.newaxis]
    mu_z = z.mean(axis=1)[..., np.newaxis]

    transfer_mat = _mkl_matrix(cov_r, cov_z)
    res = np.dot(transfer_mat, r - mu_r) + mu_z
    return res.T.reshape(src.shape)
