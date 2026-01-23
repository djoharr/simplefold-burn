use crate::blocks::{
    ConditionEmbedder, ConditionEmbedderConfig, FinalLayer, FinalLayerConfig, TimestepEmbedder,
    TimestepEmbedderConfig, Trunk, TrunkConfig,
};
use crate::data::FeaturesBatch;
use crate::utils::{absolute_position_encoding, bmm, fourier_position_encoding};
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{
    activation::{silu, softmax},
    backend::Backend,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// Folding Model ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Config, Debug)]
pub struct FoldingDiTConfig {
    h_size: usize,
    n_heads: usize,
    depth: usize,
    atom_h_size: usize,
    atom_n_heads: usize,
    atom_depth: usize,
    #[config(default = "4")]
    in_dim: usize,
    #[config(default = "256")]
    time_freq_size: usize,
    #[config(default = "2560")]
    esm_in_dim: usize,
}

#[derive(Module, Debug)]
pub struct FoldingDiT<B: Backend> {
    time_embedder: TimestepEmbedder<B>,
    atom_encoder_transformer: Trunk<B>,
    atom_decoder_transformer: Trunk<B>,
    trunk: Trunk<B>,
    atom_feat_proj_lin: Linear<B>,
    atom_feat_proj_norm: LayerNorm<B>,
    atom_pos_proj: Linear<B>,
    length_embedder_lin: Linear<B>,
    length_embedder_norm: LayerNorm<B>,
    atom_in_proj: Linear<B>,
    esm_s_proj: ConditionEmbedder<B>,
    esm_cat_proj: Linear<B>,
    context_to_atom_lin: Linear<B>,
    context_to_atom_norm: LayerNorm<B>,
    atom_to_latent_lin: Linear<B>,
    atom_to_latent_norm: LayerNorm<B>,
    atom_enc_cond_lin: Linear<B>,
    atom_enc_cond_norm: LayerNorm<B>,
    atom_dec_cond_lin: Linear<B>,
    atom_dec_cond_norm: LayerNorm<B>,
    latent_to_atom_w1: Linear<B>,
    latent_to_atom_norm: LayerNorm<B>,
    latent_to_atom_w2: Linear<B>,
    final_layer: FinalLayer<B>,
    esm_s_combine: Tensor<B, 1>,
    h_size: usize,
}

impl FoldingDiTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FoldingDiT<B> {
        FoldingDiT {
            time_embedder: TimestepEmbedderConfig::new(self.time_freq_size, self.h_size)
                .init(device),
            atom_encoder_transformer: TrunkConfig::new(
                self.atom_h_size,
                self.atom_n_heads,
                self.in_dim,
                self.atom_depth,
            )
            .init(device),
            atom_decoder_transformer: TrunkConfig::new(
                self.atom_h_size,
                self.atom_n_heads,
                self.in_dim,
                self.atom_depth,
            )
            .init(device),
            trunk: TrunkConfig::new(self.h_size, self.n_heads, self.in_dim, self.depth)
                .init(device),
            atom_feat_proj_lin: LinearConfig::new(1199 + self.h_size, self.h_size).init(device),
            atom_feat_proj_norm: LayerNormConfig::new(self.h_size).init(device),
            atom_pos_proj: LinearConfig::new(771, self.h_size)
                .with_bias(false)
                .init(device),
            length_embedder_lin: LinearConfig::new(1, self.h_size)
                .with_bias(false)
                .init(device),
            length_embedder_norm: LayerNormConfig::new(self.h_size).init(device),
            atom_in_proj: LinearConfig::new(2 * self.h_size, self.h_size)
                .with_bias(false)
                .init(device),
            esm_s_proj: ConditionEmbedderConfig::new(self.esm_in_dim, self.h_size).init(device),
            esm_cat_proj: LinearConfig::new(2 * self.h_size, self.h_size).init(device),
            context_to_atom_lin: LinearConfig::new(self.h_size, self.atom_h_size).init(device),
            context_to_atom_norm: LayerNormConfig::new(self.atom_h_size).init(device),
            atom_to_latent_lin: LinearConfig::new(self.atom_h_size, self.h_size).init(device),
            atom_to_latent_norm: LayerNormConfig::new(self.h_size).init(device),
            atom_enc_cond_lin: LinearConfig::new(self.h_size, self.atom_h_size).init(device),
            atom_enc_cond_norm: LayerNormConfig::new(self.atom_h_size).init(device),
            atom_dec_cond_lin: LinearConfig::new(self.h_size, self.atom_h_size).init(device),
            atom_dec_cond_norm: LayerNormConfig::new(self.atom_h_size).init(device),
            latent_to_atom_w1: LinearConfig::new(self.h_size, self.h_size).init(device),
            latent_to_atom_norm: LayerNormConfig::new(self.h_size).init(device),
            latent_to_atom_w2: LinearConfig::new(self.h_size, self.atom_h_size).init(device),
            final_layer: FinalLayerConfig::new(self.atom_h_size, 3, self.h_size).init(device),
            esm_s_combine: Tensor::zeros([37], device),
            h_size: self.h_size, // For absolute positional encoding
        }
    }
}

impl<B: Backend> FoldingDiT<B> {
    pub fn forward(
        &self,
        feats: FeaturesBatch<B>,
        noised_pos: Tensor<B, 3>,
        t: Tensor<B, 1>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Setup:
        let ref_pos = feats.ref_pos;
        let [b, n, _] = ref_pos.clone().dims();
        // let m = feats.mol_type.dims()[1];
        let atom_to_token = feats.atom_to_token;

        // Time:
        let length: Tensor<B, 2> = feats.max_num_tokens.float();
        let length = length.log();
        let l_emb = self.length_embedder_lin.forward(length);
        let l_emb = self.length_embedder_norm.forward(l_emb);
        let c_emb = self.time_embedder.forward(t, device);
        let c_emb = c_emb + l_emb;

        // Atom features:
        let mol_type: Tensor<B, 3> = feats.mol_type.float().one_hot(4);
        let res_feat = Tensor::cat(vec![mol_type, feats.res_type, feats.pocket_features], 2);
        let atom_feat_from_res = bmm(atom_to_token.clone(), res_feat, device);
        let atom_to_token_idx = feats.atom_to_token_idx.clone().unsqueeze_dim(2);
        let atom_ref_pos = absolute_position_encoding(atom_to_token_idx, self.h_size, device);
        let ref_pos_emb = fourier_position_encoding(ref_pos.clone(), device);
        let atom_feats = Tensor::cat(
            vec![
                ref_pos_emb,
                atom_feat_from_res,
                atom_ref_pos,
                feats.ref_charges.unsqueeze_dim(2),
                feats.atom_pad_mask.unsqueeze_dim(2),
                feats.ref_elements,
                feats.ref_atom_name_chars.reshape([b, n, 256]).float(),
            ],
            2,
        );
        let atom_feats = self.atom_feat_proj_lin.forward(atom_feats);
        let atom_feats = self.atom_feat_proj_norm.forward(atom_feats);
        let atom_coords = fourier_position_encoding(noised_pos, device);
        let atom_coords = self.atom_pos_proj.forward(atom_coords);
        let atom_in = Tensor::cat(vec![atom_feats, atom_coords], 2);
        let atom_in = self.atom_in_proj.forward(atom_in);

        // Position embeddings
        let ref_space_uid = feats.atom_to_token_idx.clone().unsqueeze_dim(2);
        let residue_id: Tensor<B, 3> = feats.residue_index.unsqueeze_dim(2).float();
        let entity_id = feats.entity_id.unsqueeze_dim(2).float();
        let asym_id = feats.asym_id.unsqueeze_dim(2).float();
        let sym_id = feats.sym_id.unsqueeze_dim(2).float();
        let atom_pe_pos = Tensor::cat(vec![ref_space_uid, ref_pos], 2);
        let token_pe_pos = Tensor::cat(vec![residue_id, entity_id, asym_id, sym_id], 2);

        // Atom encoding
        let c_emb_enc = self.atom_enc_cond_lin.forward(c_emb.clone());
        let c_emb_enc = self.atom_enc_cond_norm.forward(c_emb_enc);
        let atom_latent = self.context_to_atom_lin.forward(atom_in);
        let atom_latent = self.context_to_atom_norm.forward(atom_latent);
        let atom_latent = self.atom_encoder_transformer.forward(
            atom_latent,
            c_emb_enc,
            atom_pe_pos.clone(),
            device,
        );
        let atom_latent = self.atom_to_latent_lin.forward(atom_latent);
        let atom_latent = self.atom_to_latent_norm.forward(atom_latent);

        // ESM
        let esm_s = softmax(self.esm_s_combine.clone(), 0)
            .reshape([1, 1, 1, 37])
            .matmul(feats.esm_s)
            .squeeze_dim(2);
        let esm_emb = self.esm_s_proj.forward(esm_s);

        // Grouping and cating
        let atom_to_token_mean = atom_to_token.clone() / (atom_to_token.clone().sum_dim(1) + 1e-6);
        let latent = bmm(
            atom_to_token_mean.permute([0, 2, 1]),
            atom_latent.clone(),
            device,
        );
        let latent = self
            .esm_cat_proj
            .forward(Tensor::cat(vec![latent, esm_emb], 2));

        // Residue trunk
        let latent = self
            .trunk
            .forward(latent, c_emb.clone(), token_pe_pos, device);

        // Ungrouping + skip connection
        let output = bmm(atom_to_token, latent.clone(), device);
        let output = output + atom_latent;
        let output = self.latent_to_atom_w1.forward(output);
        let output = self.latent_to_atom_norm.forward(silu(output));
        let output = self.latent_to_atom_w2.forward(output);

        // Atom decoding
        let c_emb_dec = self.atom_dec_cond_lin.forward(c_emb.clone());
        let c_emb_dec = self.atom_dec_cond_norm.forward(c_emb_dec);
        let output = self
            .atom_decoder_transformer
            .forward(output, c_emb_dec, atom_pe_pos, device);

        // Final layer
        let predicted_velocity = self.final_layer.forward(output, c_emb);

        return (predicted_velocity, latent);
    }
}
