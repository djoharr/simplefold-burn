use crate::utils::{axial_rope, simple_layer_norm};
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{
    activation::{silu, softmax},
    backend::Backend,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// DiT ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct DiTBlockConfig {
    h_size: usize,
    n_heads: usize,
    in_dim: usize,
}

#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    // norm1: LayerNorm<B>,
    attn: SelfAttention<B>,
    // norm2: LayerNorm<B>,
    swiglu: SwiGLU<B>,
    ada_ln_modulation: Linear<B>,
}

impl DiTBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiTBlock<B> {
        let mlp_hidden_dim = self.h_size * 4;
        DiTBlock {
            // norm1: LayerNormConfig::new(self.h_size)
            //     .with_bias(false)
            //     .with_epsilon(1e-6)
            //     .init(device),
            attn: SelfAttentionConfig::new(self.h_size, self.n_heads, self.in_dim).init(device),
            // norm2: LayerNormConfig::new(self.h_size)
            //     .with_bias(false)
            //     .with_epsilon(1e-6)
            //     .init(device),
            swiglu: SwiGLUConfig::new(self.h_size, mlp_hidden_dim).init(device),
            ada_ln_modulation: LinearConfig::new(self.h_size, 6 * self.h_size).init(device),
        }
    }
}

impl<B: Backend> DiTBlock<B> {
    pub fn forward(
        &self,
        latents: Tensor<B, 3>,
        c: Tensor<B, 2>,
        pos: Tensor<B, 3>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut tensors = self.ada_ln_modulation.forward(silu(c)).chunk(6, 1);
        let shift_msa = tensors.remove(0).unsqueeze();
        let scale_msa = tensors.remove(0).unsqueeze();
        let gate_msa = tensors.remove(0).unsqueeze();
        let shift_mlp = tensors.remove(0).unsqueeze();
        let scale_mlp = tensors.remove(0).unsqueeze();
        let gate_mlp = tensors.remove(0).unsqueeze();
        // let latents_msa = self.norm1.forward(latents.clone()) * (1 + scale_msa) + shift_msa;
        let latents_msa = simple_layer_norm(latents.clone(), 1e-6) * (1 + scale_msa) + shift_msa;
        let latents_msa = self.attn.forward(latents_msa, pos, device);
        // let latents_mlp = self.norm2.forward(latents.clone()) * (1 + scale_mlp) + shift_mlp;
        let latents_mlp = simple_layer_norm(latents.clone(), 1e-6) * (1 + scale_mlp) + shift_mlp;
        let latents_mlp = self.swiglu.forward(latents_mlp);
        let latents = latents.clone() + gate_msa * latents_msa + gate_mlp * latents_mlp;
        return latents;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct TrunkConfig {
    h_size: usize,
    n_heads: usize,
    in_dim: usize,
    depth: usize,
}

#[derive(Module, Debug)]
pub struct Trunk<B: Backend> {
    layers: Vec<DiTBlock<B>>,
}

impl TrunkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Trunk<B> {
        let mut layers = Vec::new();
        for _ in 0..self.depth {
            let layer = DiTBlockConfig::new(self.h_size, self.n_heads, self.in_dim).init(device);
            layers.push(layer)
        }
        Trunk { layers: layers }
    }
}

impl<B: Backend> Trunk<B> {
    pub fn forward(
        &self,
        latents: Tensor<B, 3>,
        c: Tensor<B, 2>,
        pos: Tensor<B, 3>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut x = latents;
        for layer in &self.layers {
            x = layer.forward(x, c.clone(), pos.clone(), device);
        }
        return x;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// ATTENTION BLOCK //////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct SelfAttentionConfig {
    hidden_size: usize,
    num_heads: usize,
    in_dim: usize,
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    in_dim: usize,
    n_heads: usize,
    h_size: usize,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        let head_dim = self.hidden_size / self.num_heads;
        SelfAttention {
            qkv: LinearConfig::new(self.hidden_size, self.hidden_size * 3)
                .with_bias(false)
                .init(device),
            proj: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            q_norm: RmsNormConfig::new(head_dim).init(device),
            k_norm: RmsNormConfig::new(head_dim).init(device),
            in_dim: self.in_dim,
            n_heads: self.num_heads,
            h_size: self.hidden_size,
        }
    }
}

impl<B: Backend> SelfAttention<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, n, c] = input.dims();
        let qkv = self
            .qkv
            .forward(input)
            .reshape([b, n, 3, self.n_heads, c / self.n_heads])
            .permute([2, 0, 3, 1, 4]);

        let q: Tensor<B, 4> = qkv.clone().slice(s![0, .., .., .., ..]).squeeze_dim::<4>(0);
        let k: Tensor<B, 4> = qkv.clone().slice(s![1, .., .., .., ..]).squeeze_dim::<4>(0);
        let v: Tensor<B, 4> = qkv.clone().slice(s![2, .., .., .., ..]).squeeze_dim::<4>(0);
        let (q, k) = axial_rope::<B>(q, k, pos, self.in_dim, self.n_heads, self.h_size, &device);
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let scale = (c as f64 / self.n_heads as f64).powf(-0.25);
        let k = k.transpose();
        let attn = q.matmul(k) * scale;
        let attn = softmax(attn, 3);
        let out = attn.matmul(v).swap_dims(1, 2).flatten(2, 3);

        self.proj.forward(out)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// EMBEDDERS /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct ConditionEmbedderConfig {
    input_dim: usize,
    h_size: usize,
}

#[derive(Module, Debug)]
pub struct ConditionEmbedder<B: Backend> {
    linear: Linear<B>,
    norm: LayerNorm<B>,
}

impl ConditionEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConditionEmbedder<B> {
        ConditionEmbedder {
            linear: LinearConfig::new(self.input_dim, self.h_size).init(device),
            norm: LayerNormConfig::new(self.h_size).init(device),
        }
    }
}

impl<B: Backend> ConditionEmbedder<B> {
    pub fn forward(&self, cond: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear.forward(cond);
        let x = self.norm.forward(x);
        let x = silu(x);
        return x;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct TimestepEmbedderConfig {
    freq_emb_size: usize,
    h_size: usize,
}

#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    freq_emb_size: usize,
}

impl TimestepEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
        TimestepEmbedder {
            linear1: LinearConfig::new(self.freq_emb_size, self.h_size).init(device),
            linear2: LinearConfig::new(self.h_size, self.h_size).init(device),
            freq_emb_size: self.freq_emb_size,
        }
    }
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn forward(&self, t: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 2> {
        let half = (self.freq_emb_size / 2) as i64;
        let max_period: Tensor<B, 1> = -Tensor::from_floats([10000], device).log() / half;
        let freqs: Tensor<B, 1> = (max_period * Tensor::arange(0..half, device).float()).exp();
        let coefs = t.unsqueeze::<2>() * freqs.unsqueeze::<2>();
        let mut embeddings = Tensor::cat(vec![coefs.clone().cos(), coefs.sin()], 1);
        if self.freq_emb_size % 2 != 0 {
            embeddings = Tensor::cat(vec![embeddings, Tensor::zeros([1, 1], device)], 1);
        }
        let x = self.linear1.forward(embeddings);
        let x = silu(x);
        let x = self.linear2.forward(x);
        return x;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// OTHER LAYERS ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct SwiGLUConfig {
    dim: usize,
    h_dim: usize,
}

#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}

impl SwiGLUConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGLU<B> {
        let h_size = 256 * (((2 * self.h_dim / 3) + 256 - 1) / 256);
        SwiGLU {
            w1: LinearConfig::new(self.dim, h_size)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(h_size, self.dim).init(device),
            w3: LinearConfig::new(self.dim, h_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> SwiGLU<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = silu(self.w1.forward(x.clone()));
        let x2 = self.w3.forward(x);
        self.w2.forward(x1 * x2)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct FinalLayerConfig {
    h_size: usize,
    out_channels: usize,
    c_dim: usize,
}

#[derive(Module, Debug)]
pub struct FinalLayer<B: Backend> {
    ada_ln_modulation: Linear<B>,
    // norm_final: LayerNorm<B>,
    linear: Linear<B>,
}

impl FinalLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FinalLayer<B> {
        FinalLayer {
            ada_ln_modulation: LinearConfig::new(self.c_dim, 2 * self.h_size).init(device),
            // norm_final: LayerNormConfig::new(self.h_size)
            //     .with_bias(false)
            //     .with_epsilon(1e-6)
            //     .init(device),
            linear: LinearConfig::new(self.h_size, self.out_channels).init(device),
        }
    }
}

impl<B: Backend> FinalLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 2>) -> Tensor<B, 3> {
        let mut tensors = self.ada_ln_modulation.forward(silu(c)).chunk(2, 1);
        let shift = tensors.remove(0).unsqueeze();
        let scale = tensors.remove(0).unsqueeze();
        let x = simple_layer_norm(x, 1e-6) * (1 + scale) + shift;
        // let x = self.norm_final.forward(x) * (1 + scale) + shift;
        self.linear.forward(x)
    }
}
