use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::tensor::linalg;
use std::cmp;
use std::f64::consts::PI;

pub fn complex_mul<B: Backend>(x: Tensor<B, 5>, y: Tensor<B, 5>) -> Tensor<B, 5> {
    let xr = x.clone().slice(s![.., .., .., .., 0]);
    let xi = x.slice(s![.., .., .., .., 1]);
    let yr = y.clone().slice(s![.., .., .., .., 0]);
    let yi = y.slice(s![.., .., .., .., 1]);
    let real = xr.clone() * yr.clone() - xi.clone() * yi.clone();
    let imag = xr * yi + xi * yr;
    Tensor::cat(vec![real, imag], 4)
}

pub fn linspace(start: &f64, end: &f64, n_steps: &i32, include_end: &bool) -> Vec<f64> {
    let interval: f64 = end - start;
    let step_size = if *include_end {
        interval / (*n_steps as f64 - 1.0)
    } else {
        interval / (*n_steps as f64)
    };
    let mut line: Vec<f64> = Vec::new();
    for n in 0..*n_steps {
        line.push(start + step_size * n as f64);
    }
    return line;
}

pub fn kron_mul<B: Backend>(m: Tensor<B, 2>, v: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 3> {
    let d = m.dims();
    let mut result = Tensor::<B, 3>::empty([d[0], d[1], v.dims()[0]], device);
    for i in 0..d[0] {
        for j in 0..d[1] {
            let coef = m.clone().slice([i, j]).squeeze_dim::<1>(0);
            result = result.slice_assign(s![i, j, ..], (coef * v.clone()).unsqueeze::<3>());
        }
    }
    result
}

pub fn axial_rope<B: Backend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    pos: Tensor<B, 3>,
    in_dim: usize,
    n_heads: usize,
    embed_dim: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let interval = 2 * in_dim;
    let true_embed_dim = (embed_dim / n_heads) as i64;

    let base_freq: Tensor<B, 1> = Tensor::arange_step(0..true_embed_dim, interval, device).float();
    let theta: Tensor<B, 1> = Tensor::from_floats([100], device);
    let freq: Tensor<B, 1> = theta.powf(-base_freq / true_embed_dim);
    let mut all_freqs_ = Tensor::<B, 3>::empty([pos.dims()[1], freq.dims()[0] * 4, 2], device); // Might not work in batches

    for i in 0..in_dim {
        let t: Tensor<B, 1> = pos.clone().slice(s![.., .., i]).flatten(0, 2);
        let freq_i: Tensor<B, 2> = linalg::outer(t, freq.clone());
        let freq_ic: Tensor<B, 2> = freq_i.clone().cos();
        let freq_is: Tensor<B, 2> = freq_i.clone().sin();
        let freq_ip = Tensor::stack::<3>(vec![freq_ic, freq_is], 2);
        all_freqs_ = all_freqs_.slice_assign(s![.., 8 * i..8 * i + 8, ..], freq_ip);
    }

    let all_freqs = all_freqs_.unsqueeze::<5>();
    let [d1, d2, d3, _] = xq.dims();

    let mut xq_ = xq.reshape([d1 as i32, d2 as i32, d3 as i32, -1, 2]);
    xq_ = complex_mul(xq_, all_freqs.clone());
    let xq_out: Tensor<B, 4> = xq_.flatten(3, 4);

    let mut xk_ = xk.reshape([d1 as i32, d2 as i32, d3 as i32, -1, 2]);
    xk_ = complex_mul(xk_, all_freqs.clone());
    let xk_out: Tensor<B, 4> = xk_.flatten(3, 4);

    return (xq_out, xk_out);
}

pub fn absolute_position_encoding<B: Backend>(
    pos: Tensor<B, 3>,
    embed_dim: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let h_dim = (embed_dim / 2) as i32;
    let end = (224 as f64).log2() - 1.;
    let ls = linspace(&0., &end, &h_dim, &true);
    let omega = Tensor::<B, 1>::from_data(TensorData::new(ls.clone(), [ls.len()]), &device);
    let omega: Tensor<B, 1> = (Tensor::from_floats([2], device).powf(omega)) * PI;
    let emb = kron_mul(pos.clone().squeeze_dim::<2>(2), omega.clone(), device);
    let embs = Tensor::cat(vec![emb.clone().sin(), emb.clone().cos()], 2);
    Tensor::cat(vec![embs, pos], 2)
}

pub fn fourier_position_encoding<B: Backend>(
    pos: Tensor<B, 3>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let ls = linspace(&0., &12., &128, &true);
    let ls_tensor = Tensor::<B, 1>::from_data(TensorData::new(ls.clone(), [ls.len()]), &device);
    let freq_bands = Tensor::from_floats([2], device).powf(ls_tensor);

    let d = pos.dims();
    let mut embs = Tensor::<B, 4>::empty([d[0], d[1], d[2], freq_bands.dims()[0]], device);
    for i in 0..d[0] {
        let m = pos.clone().slice(s![i, .., ..]).squeeze_dim::<2>(0);
        embs = embs.slice_assign(
            s![i, .., .., ..],
            kron_mul(m, freq_bands.clone(), device).unsqueeze::<4>(),
        );
    }
    let embs_sin: Tensor<B, 3> = embs.clone().sin().flatten(2, 3);
    let embs_cos: Tensor<B, 3> = embs.clone().cos().flatten(2, 3);

    Tensor::cat(vec![pos, embs_sin, embs_cos], 2)
}

pub fn create_attn_mask<B: Backend>(
    n: usize,
    n_queries: usize,
    n_keys: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n_trunks = n / n_queries; // Maybe should ceil here
    let n_padded = n_trunks * n_queries;
    let mut attn_mask: Tensor<B, 2> = Tensor::zeros([n_padded, n_padded], device);
    for block_idx in 0..n_trunks {
        let i = block_idx * n_queries;
        let j1 = cmp::max(0, n_queries * block_idx - (n_keys - n_queries) / 2);
        let j2 = n_queries * block_idx + (n_keys + n_queries) / 2;
        attn_mask = attn_mask.slice_fill([i..i + n_queries, j1..j2], 1.0);
    }
    let attn_bias: Tensor<B, 2> = (1 - attn_mask) * -1e10;
    let attn_bias = attn_bias.slice([0..n, 0..n]);
    return attn_bias;
}

pub fn bmm<B: Backend>(
    batch_mat_1: Tensor<B, 3>,
    batch_mat_2: Tensor<B, 3>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [b, n, _] = batch_mat_1.dims(); // b, n, m
    let [_, _, p] = batch_mat_2.dims(); // b, m, p
    let mut out = Tensor::<B, 3>::empty([b, n, p], device);
    for i in 0..b {
        let m1 = batch_mat_1.clone().slice(s![i, .., ..]);
        let m2 = batch_mat_2.clone().slice(s![i, .., ..]);
        out = out.slice_assign(s![i, .., ..], m1.matmul(m2));
    }
    return out;
}

pub fn simple_layer_norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    let (var, mean) = input.clone().var_mean_bias(D - 1);
    let output = input.sub(mean).div(var.add_scalar(epsilon).sqrt());
    return output;
}

pub fn center_shape<B: Backend>(coords: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
    let coords = coords.clone();
    let mask = mask.unsqueeze_dim(2);
    let mean = coords.clone() * mask.clone();
    let mean = mean.sum_dim(1) / mask.sum_dim(1);
    let coords = coords - mean;
    return coords;
}
