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
    let args: Vec<String> = std::env::args().collect();
    let model_name = &args[1];
    let seq = &args[2];

    type B = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    B::seed(&device, 42);

    let ccd_path = "data/ccd.json";
    let features = process_data::<B>(ccd_path, seq, 32, device.clone());

    // Flow config
    let num_timesteps = 100;
    let tau = 0.05;
    let t_start = 1e-4;
    let w_cutoff = 0.99;

    let config_path = format!("data/configs/{}.json", model_name);
    let loaded_config = FoldingDiTConfig::load(config_path).unwrap();
    let record = get_model_record(model_name, &"data", &device);
    let model = loaded_config.init::<B>(&device).load_record(record);

    let flow = Flow {
        num_timesteps,
        t_start,
        tau,
        w_cutoff,
    };

    println!("Starting forward pass");
    let distribution = Distribution::Normal(0.0, 1.0);
    let noised_pos = Tensor::<B, 3>::random(features.coords.shape(), distribution, &device);
    let res = flow.sample(model, noised_pos, features, &device);
    println!("Results: {res}");
    println!("Done !");
}
