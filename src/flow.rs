use crate::data::FeaturesBatch;
use crate::model::FoldingDiT;
use crate::utils::{center_shape, linspace};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::Backend;
use kdam::tqdm;

pub struct Flow {
    pub num_timesteps: i32,
    pub t_start: f32,
    pub tau: f32,
    pub w_cutoff: f32,
}

impl Flow {
    pub fn euler_maruyama_step<B: Backend>(
        &self,
        model: FoldingDiT<B>,
        y_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        t_next: Tensor<B, 1>,
        feats: FeaturesBatch<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let dt = t_next - t.clone();
        let distribution = Distribution::Uniform(0.0, 1.0);
        let eps = y_t.random_like(distribution);
        let y = center_shape(y_t, feats.clone().atom_pad_mask);
        let batched_t = t.clone().repeat(&[y.dims()[0]]);

        let (velocity, _) = model.forward(feats, y.clone(), batched_t, device);
        let score = compute_linear_flow(velocity.clone(), y.clone(), t.clone());

        let t: f32 = t.into_data().to_vec().unwrap()[0];
        let mut diff_coeff = (1.0 - t) / (t + 0.01);
        if t >= self.w_cutoff {
            diff_coeff = 0.0
        }
        let drift = velocity + diff_coeff * score;
        let mean_y = y + drift * dt.clone().into_scalar();
        let shift: Tensor<B, 1> = 2.0 * diff_coeff * self.tau * dt;
        let y_sample = mean_y + eps * shift.sqrt().into_scalar();

        return y_sample;
    }

    pub fn sample<B: Backend>(
        &self,
        model: FoldingDiT<B>,
        noise: Tensor<B, 3>,
        feats: FeaturesBatch<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let n_steps = self.num_timesteps + 1;
        let ls = linspace(&-2., &0., &n_steps, &true);
        let ls_tensor = Tensor::<B, 1>::from_data(TensorData::new(ls.clone(), [ls.len()]), &device);
        let steps = Tensor::from_floats([2], device).powf(ls_tensor);
        let steps = steps.clone() - steps.min();
        let steps = steps.clone() / steps.max();
        let steps = steps.clamp(self.t_start, 1.0);

        let mut y_t = noise;
        for i in tqdm!(0..self.num_timesteps) {
            let t = steps.clone().slice(i);
            let t_next = steps.clone().slice(i + 1);
            y_t = self.euler_maruyama_step(model.clone(), y_t, t, t_next, feats.clone(), device)
        }

        return y_t;
    }
}

pub fn compute_linear_flow<B: Backend>(
    v_t: Tensor<B, 3>,
    y_t: Tensor<B, 3>,
    t: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let t: Tensor<B, 3> = t.unsqueeze_dims(&[-1, -1]);
    let alpha_t = t.clone();
    let d_alpha_t = 1;
    let sigma_t: Tensor<B, 3> = 1 - t;
    let d_sigma_t = -1;

    let inv_ratio = alpha_t / d_alpha_t;
    let var = sigma_t.clone().powf_scalar(2) - inv_ratio.clone() * d_sigma_t * sigma_t;
    let score = (inv_ratio * v_t - y_t) / var;
    return score;
}
