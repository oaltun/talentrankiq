//! Pooling and L2-normalization, op-for-op the same sequence as mlxcel's
//! `pool` / `normalize_l2` so a parity test compares *models*, not two
//! slightly different pooling implementations.
//!
//! The masking detail that matters: the real-token set comes from the
//! `[B, L]` padding mask, never from "position L-1" — a right-padded batch's
//! last position is padding.

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::checkpoint::Pooling;

/// `[B, L, D]` hidden + `[B, L]` mask (1 = real) -> `[B, D]` f32.
pub fn pool(hidden: &MlxArray, mask: &MlxArray, mode: Pooling) -> UniquePtr<MlxArray> {
    let hidden = mlxcel_core::astype(hidden, mlxcel_core::dtype::FLOAT32);
    let shape = mlxcel_core::array_shape(&hidden);
    assert!(
        shape.len() == 3,
        "pool: hidden must be [B, L, D], got {shape:?}"
    );
    let (b, l, d) = (shape[0], shape[1], shape[2]);
    let zero_i32 = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
    let real = mlxcel_core::not_equal(mask, &zero_i32);

    let pooled = match mode {
        Pooling::Mean => {
            let mask_f = mlxcel_core::astype(&real, mlxcel_core::dtype::FLOAT32);
            let weighted =
                mlxcel_core::multiply(&hidden, &mlxcel_core::reshape(&mask_f, &[b, l, 1]));
            let summed = mlxcel_core::sum_axis(&weighted, 1, false);
            let counts = mlxcel_core::sum_axis(&mask_f, 1, true);
            let denom =
                mlxcel_core::maximum(&counts, &mlxcel_core::from_slice_f32(&[1e-9], &[1, 1]));
            mlxcel_core::divide(&summed, &denom)
        }
        Pooling::Cls => {
            let first = mlxcel_core::argmax(&real, 1, true);
            gather_rows(&hidden, &first, b, d)
        }
        Pooling::LastToken | Pooling::Default => {
            // position index of each real token, -1 for padding; the max is
            // the last real position, and an all-pad row falls back to L-1 so
            // the gather stays in bounds (its output is never read).
            let positions = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, l, 1), &[1, l]);
            let none = mlxcel_core::from_slice_i32(&[-1], &[1, 1]);
            let candidates = mlxcel_core::where_cond(&real, &positions, &none);
            let last = mlxcel_core::max_axis(&candidates, 1, true);
            let fallback = mlxcel_core::from_slice_i32(&[l - 1], &[1, 1]);
            let last =
                mlxcel_core::where_cond(&mlxcel_core::less(&last, &zero_i32), &fallback, &last);
            gather_rows(&hidden, &last, b, d)
        }
    };
    mlxcel_core::reshape(&pooled, &[b, d])
}

fn gather_rows(hidden: &MlxArray, index: &MlxArray, b: i32, d: i32) -> UniquePtr<MlxArray> {
    let index = mlxcel_core::reshape(index, &[b, 1, 1]);
    let picked = mlxcel_core::take_along_axis(hidden, &index, 1);
    mlxcel_core::reshape(&picked, &[b, d])
}

/// `x / max(||x||_2, 1e-9)` over the last axis — the same `EmbdNormalize`
/// euclidean path the reference runtime uses, so a zero row stays zero
/// instead of becoming NaN.
pub fn normalize_l2(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let norm = mlxcel_core::linalg_norm(&x, -1, true);
    let denom = mlxcel_core::maximum(&norm, &mlxcel_core::from_slice_f32(&[1e-9], &[1, 1]));
    mlxcel_core::divide(&x, &denom)
}

/// Read a `[rows, dim]` f32 array back to the host.
pub fn to_host(x: &MlxArray) -> Vec<Vec<f32>> {
    mlxcel_core::try_eval(x).expect("mlx eval");
    let shape = mlxcel_core::array_shape(x);
    let (rows, dim) = (shape[0] as usize, shape[1] as usize);
    let flat = array_to_vec_f32(x);
    (0..rows)
        .map(|r| flat[r * dim..(r + 1) * dim].to_vec())
        .collect()
}