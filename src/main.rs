#![recursion_limit = "256"]
use crate::data::{get_model_record, process_data};
use crate::flow::Flow;
use crate::model::FoldingDiTConfig;
use burn::backend::wgpu::Wgpu;
use burn::prelude::*;
use burn::tensor::Distribution;
mod blocks;
mod data;
mod flow;
mod model;
mod ref_constants;
mod utils;

fn main() {
    let seq = "AG";
    let feats = data::get_seq_info("data/ccd.json".to_string(), seq.to_string()).unwrap();
    let f = data::get_vecs_from_feats(feats);
    println!("{:?}", f.atoms);

    type B = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    B::seed(&device, 42);

    let features = process_data::<B>(
        "data/ccd.json".to_string(),
        seq.to_string(),
        32,
        device.clone(),
    );

    // 100M Configs:
    let h_size = 768;
    let n_heads = 12;
    let depth = 8;
    let atom_h_size = 256;
    let atom_n_heads = 4;
    let atom_depth = 1;

    // Flow config
    let num_timesteps = 1000;
    let tau = 0.05;
    let t_start = 1e-4;
    let w_cutoff = 0.99;

    let record = get_model_record(&"simplefold_100M", &"data", &device);

    let model = FoldingDiTConfig::new(
        h_size,
        n_heads,
        depth,
        atom_h_size,
        atom_n_heads,
        atom_depth,
    )
    .init::<B>(&device)
    .load_record(record);

    let flow = Flow {
        num_timesteps,
        t_start,
        tau,
        w_cutoff,
    };

    let distribution = Distribution::Normal(0.0, 1.0);
    let noised_pos = Tensor::<B, 3>::random(features.coords.shape(), distribution, &device);

    println!("Starting forward pass");
    let res = flow.sample(model, noised_pos, features, &device);
    println!("Done !");
}
