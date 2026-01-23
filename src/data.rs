use crate::model::FoldingDiTRecord;
use crate::ref_constants::get_all_refs;
use reqwest::blocking::get;
use std::{error, fs, io::Write, path::Path, time::Instant};
// use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use polars::prelude::*;
use pyo3::ffi::c_str;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub struct BaseFeatures {
    pub atoms: Vec<Option<String>>,
    pub elements: Vec<Option<i64>>,
    pub charges: Vec<Option<i64>>,
    pub confx: Vec<Option<f64>>,
    pub confy: Vec<Option<f64>>,
    pub confz: Vec<Option<f64>>,
}
#[derive(Clone)]
pub struct VecFeatures {
    pub atoms: Vec<String>,
    pub elements: Vec<i64>,
    pub charges: Vec<i64>,
    pub confx: Vec<f64>,
    pub confy: Vec<f64>,
    pub confz: Vec<f64>,
}

pub fn get_seq_info(cache: String, aa: String) -> Result<BaseFeatures, PolarsError> {
    let mut file = fs::File::open(cache).unwrap();
    let ccd = match JsonReader::new(&mut file).finish() {
        Ok(lf) => lf,
        Err(e) => panic!("Error: {}", e),
    };

    let mut df = DataFrame::empty_with_schema(ccd.schema());
    for el in aa.chars() {
        let el_info = ccd
            .clone()
            .lazy()
            .filter(col("aa1").eq(lit(el.to_string())))
            .collect();
        df = concat([df.lazy(), el_info?.lazy()], UnionArgs::default())?.collect()?;
    }

    let atoms: Vec<Option<String>> = df["atoms"]
        .str()?
        .into_iter()
        .map(|s| s.map(|s| s.to_string()))
        .collect();
    let elements: Vec<Option<i64>> = df["elements"].i64()?.into_iter().collect();
    let charges: Vec<Option<i64>> = df["charges"].i64()?.into_iter().collect();
    let confx: Vec<Option<f64>> = df["confx"].f64()?.into_iter().collect();
    let confy: Vec<Option<f64>> = df["confy"].f64()?.into_iter().collect();
    let confz: Vec<Option<f64>> = df["confz"].f64()?.into_iter().collect();

    let feats = BaseFeatures {
        atoms,
        elements,
        charges,
        confx,
        confy,
        confz,
    };
    Ok(feats)
}

pub fn get_vecs_from_feats(f: BaseFeatures) -> VecFeatures {
    let atoms: Vec<String> = f
        .atoms
        .iter()
        .map(|a| a.clone().unwrap_or_default())
        .collect();
    let elements: Vec<i64> = f.elements.iter().map(|e| e.unwrap_or_default()).collect();
    let charges: Vec<i64> = f.charges.iter().map(|c| c.unwrap_or_default()).collect();
    let confx: Vec<f64> = f.confx.iter().map(|x| x.unwrap_or_default()).collect();
    let confy: Vec<f64> = f.confy.iter().map(|y| y.unwrap_or_default()).collect();
    let confz: Vec<f64> = f.confz.iter().map(|z| z.unwrap_or_default()).collect();

    let feats = VecFeatures {
        atoms,
        elements,
        charges,
        confx,
        confy,
        confz,
    };
    return feats;
}

pub fn get_conformers<B: Backend>(
    cx: Vec<f64>,
    cy: Vec<f64>,
    cz: Vec<f64>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n = cx.len();
    let confx = Tensor::<B, 1>::from_data(TensorData::new(cx, [n]), &device);
    let confy = Tensor::<B, 1>::from_data(TensorData::new(cy, [n]), &device);
    let confz = Tensor::<B, 1>::from_data(TensorData::new(cz, [n]), &device);
    let conformers = Tensor::stack::<2>(vec![confx, confy, confz], 1);
    return conformers;
}

pub fn encode_atoms<B: Backend>(atoms: Vec<String>, device: &B::Device) -> Tensor<B, 2, Int> {
    let mut v = Vec::new();
    for el in atoms.iter() {
        let mut left = 0;
        for (j, b) in el.bytes().enumerate() {
            let ord = b - 32;
            v.push(ord as i32);
            left = j;
        }
        for _ in 0..3 - left {
            v.push(0)
        }
    }
    let n = v.len();
    let t = Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), &device);
    let reshaped = t.reshape([-1, 4]);
    return reshaped;
}

pub fn get_token_maps<B: Backend>(
    sequence: String,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let (ref_atoms, tokens, restype_1to3) = get_all_refs();
    let mut res_type = Tensor::<B, 2>::zeros([sequence.len(), tokens.len()], device);
    let mut v = Vec::new();

    for (i, el) in sequence.chars().enumerate() {
        let token_name = restype_1to3.get(&el.to_string()).unwrap();
        let idx = tokens.iter().position(|&r| r == token_name).unwrap();
        let n_atoms = ref_atoms.get(token_name).unwrap().len();
        res_type = res_type.slice_assign(s![i, idx], Tensor::<B, 2>::from_floats([[1.]], &device));
        for _ in 0..n_atoms {
            v.push(i as i64)
        }
    }
    let ref_space_uid = Tensor::<B, 1>::from_data(TensorData::new(v.clone(), [v.len()]), &device);

    return (res_type, ref_space_uid);
}

pub fn compute_esm_representation<B: Backend>(aa_seq: String, device: &B::Device) -> Tensor<B, 3> {
    let code = c_str!(include_str!("../compute_esm.py"));
    Python::initialize();
    let res: Vec<Vec<Vec<f32>>> = Python::attach(|py| {
        let esm = PyModule::from_code(py, code, c"compute_esm.py", c"compute_esm").unwrap();
        let results = esm
            .getattr("process_esm")
            .unwrap()
            .call1((&aa_seq,))
            .unwrap();
        let res: Vec<Vec<Vec<f32>>> = results.extract().unwrap();
        res
    });
    let res_flat = res.concat().concat();
    let flat_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(res_flat, [&aa_seq.len() * 37 * 2560]),
        &device,
    );
    let res_tensor = flat_tensor.reshape([aa_seq.len(), 37, 2560]); ////// NEEDED TO FLATTEN + RESHAPE -> Possible problem
    return res_tensor;
}

#[derive(Clone)]
pub struct Features<B: Backend> {
    pub num_repeats: Tensor<B, 1, Int>,
    pub residue_index: Tensor<B, 1, Int>,
    pub entity_id: Tensor<B, 1, Int>,
    pub asym_id: Tensor<B, 1, Int>,
    pub sym_id: Tensor<B, 1, Int>,
    pub mol_type: Tensor<B, 1, Int>,
    pub pocket_features: Tensor<B, 2>,
    pub ref_pos: Tensor<B, 2>,
    pub coords: Tensor<B, 2>,
    pub max_num_tokens: Tensor<B, 1, Int>,
    pub cropped_num_tokens: Tensor<B, 1, Int>,
    pub atom_pad_mask: Tensor<B, 1>,
    pub res_type: Tensor<B, 2>,
    pub ref_space_uid: Tensor<B, 1>,
    pub atom_to_token: Tensor<B, 2>,
    pub atom_to_token_idx: Tensor<B, 1>,
    pub ref_charges: Tensor<B, 1>,
    pub ref_elements: Tensor<B, 2>,
    pub ref_atom_name_chars: Tensor<B, 3, Int>,
    pub esm_s: Tensor<B, 3>,
}

pub fn process_data<B: Backend>(
    cache: String,
    aa: String,
    atoms_per_wq: i32,
    device: B::Device,
) -> FeaturesBatch<B> {
    let num_elements = 128;

    let feats = get_seq_info(cache.clone(), aa.clone()).unwrap();
    let f = get_vecs_from_feats(feats);

    let seq_len = aa.len();
    let n_len = seq_len as i64;
    let n_atoms = f.atoms.len();
    let n = n_atoms as i32;
    let pad_len = ((n - 1) / atoms_per_wq + 1) * atoms_per_wq - n;
    let pad_len = pad_len as usize;

    let atoms = encode_atoms::<B>(f.atoms, &device);
    let atom_name_chars: Tensor<B, 3, Int> = atoms.clone().one_hot(64);
    let ref_atom_name_chars = atom_name_chars
        .permute([2, 1, 0])
        .pad((0, pad_len, 0, 0), 0.0)
        .permute([2, 1, 0]);

    let elements = Tensor::<B, 1>::from_data(TensorData::new(f.elements, [n_atoms]), &device);
    let elements: Tensor<B, 2> = elements.one_hot(num_elements);
    let ref_elements = elements.clone().pad((0, 0, 0, pad_len), 0.0);

    let mut pocket_features = Tensor::<B, 2>::zeros(Shape::new([seq_len, 4]), &device);
    pocket_features = pocket_features.slice_fill(s![.., 0], 1.0);

    let charges = Tensor::<B, 1>::from_data(TensorData::new(f.charges, [n_atoms]), &device);
    let zero_pads = Tensor::<B, 1>::zeros([pad_len], &device);
    let ref_charges = Tensor::cat(vec![charges.clone(), zero_pads.clone()], 0);

    let cx = f.confx;
    let cy = f.confy;
    let cz = f.confz;
    let conformers = get_conformers::<B>(cx, cy, cz, &device);
    let ref_pos = conformers.clone().pad((0, 0, 0, pad_len), 0.0) / 5.0;

    let atom_mask = Tensor::<B, 1>::ones([n_atoms], &device);
    let atom_pad_mask = Tensor::cat(vec![atom_mask.clone(), zero_pads.clone()], 0);

    let (res_type, ref_space_uid) = get_token_maps::<B>(aa.clone(), &device);
    let atom_to_token: Tensor<B, 2> = ref_space_uid.clone().one_hot(seq_len);
    let atom_to_token_idx = Tensor::cat(vec![ref_space_uid.clone(), zero_pads.clone()], 0);
    let atom_to_token = atom_to_token.clone().pad((0, 0, 0, pad_len), 0.0);

    let coords = ref_pos.zeros_like();
    let residue_index = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_len, &device);
    let asym_id = residue_index.zeros_like();
    let entity_id = residue_index.zeros_like();
    let sym_id = residue_index.zeros_like();
    let mol_type = residue_index.zeros_like();
    let max_num_tokens = Tensor::<B, 1, burn::tensor::Int>::from_ints([n_len], &device);
    let cropped_num_tokens = Tensor::<B, 1, burn::tensor::Int>::from_ints([n_len], &device);
    let num_repeats = Tensor::<B, 1, burn::tensor::Int>::from_ints([1], &device);
    let esm_encoding = compute_esm_representation::<B>(aa.clone(), &device);

    FeaturesBatch {
        num_repeats: num_repeats.unsqueeze(),
        residue_index: residue_index.unsqueeze(),
        entity_id: entity_id.unsqueeze(),
        asym_id: asym_id.unsqueeze(),
        sym_id: sym_id.unsqueeze(),
        mol_type: mol_type.unsqueeze(),
        pocket_features: pocket_features.unsqueeze(),
        ref_pos: ref_pos.unsqueeze(),
        coords: coords.unsqueeze(),
        max_num_tokens: max_num_tokens.unsqueeze(),
        cropped_num_tokens: cropped_num_tokens.unsqueeze(),
        atom_pad_mask: atom_pad_mask.unsqueeze(),
        res_type: res_type.unsqueeze(),
        ref_space_uid: ref_space_uid.unsqueeze(),
        atom_to_token: atom_to_token.unsqueeze(),
        atom_to_token_idx: atom_to_token_idx.unsqueeze(),
        ref_charges: ref_charges.unsqueeze(),
        ref_elements: ref_elements.unsqueeze(),
        ref_atom_name_chars: ref_atom_name_chars.unsqueeze(),
        esm_s: esm_encoding.unsqueeze(),
    }

    // Features {
    //     num_repeats: num_repeats,
    //     residue_index: residue_index,
    //     entity_id: entity_id,
    //     asym_id: asym_id,
    //     sym_id: sym_id,
    //     mol_type: mol_type,
    //     pocket_features: pocket_features,
    //     ref_pos: ref_pos,
    //     coords: coords,
    //     max_num_tokens: max_num_tokens,
    //     cropped_num_tokens: cropped_num_tokens,
    //     atom_pad_mask: atom_pad_mask,
    //     res_type: res_type,
    //     ref_space_uid: ref_space_uid,
    //     atom_to_token: atom_to_token,
    //     atom_to_token_idx: atom_to_token_idx,
    //     ref_charges: ref_charges,
    //     ref_elements: ref_elements,
    //     ref_atom_name_chars: ref_atom_name_chars,
    //     esm_s: esm_encoding,
    // }
}

#[derive(Clone)]
pub struct FeaturesBatch<B: Backend> {
    pub num_repeats: Tensor<B, 2, Int>,
    pub residue_index: Tensor<B, 2, Int>,
    pub entity_id: Tensor<B, 2, Int>,
    pub asym_id: Tensor<B, 2, Int>,
    pub sym_id: Tensor<B, 2, Int>,
    pub mol_type: Tensor<B, 2, Int>,
    pub pocket_features: Tensor<B, 3>,
    pub ref_pos: Tensor<B, 3>,
    pub coords: Tensor<B, 3>,
    pub max_num_tokens: Tensor<B, 2, Int>,
    pub cropped_num_tokens: Tensor<B, 2, Int>,
    pub atom_pad_mask: Tensor<B, 2>,
    pub res_type: Tensor<B, 3>,
    pub ref_space_uid: Tensor<B, 2>,
    pub atom_to_token: Tensor<B, 3>,
    pub atom_to_token_idx: Tensor<B, 2>,
    pub ref_charges: Tensor<B, 2>,
    pub ref_elements: Tensor<B, 3>,
    pub ref_atom_name_chars: Tensor<B, 4, Int>,
    pub esm_s: Tensor<B, 4>,
}

// TODO: IMPLEMENT BATCH LOGIC
// #[derive(Clone, Default)]
// pub struct FeaturesBatcher {}
//
// impl<B: Backend> Batcher<B, Features, FeaturesBatch<B>> for FeaturesBatcher {
//     fn batch(&self, items: Vec<Features>, device: &B::Device) -> FeaturesBatch<B> {}
// }

fn linear_download(download_url: &str, file_name: &str) -> Result<(), Box<dyn error::Error>> {
    let now = Instant::now();
    let response = get(download_url)?;
    let content = response.bytes()?;

    let mut downloaded_file = fs::File::create(file_name)?;
    downloaded_file.write_all(&content)?;

    let duration = now.elapsed();
    println!("Downloaded file in {duration:?}");
    Ok(())
}

pub fn load_weights<B: Backend>(model_path: String, device: &B::Device) -> FoldingDiTRecord<B> {
    let load_args = LoadArgs::new(model_path.into()) // "weights/simplefold_100M.ckpt"
        .with_key_remap(r"time_embedder.mlp.0\.", r"time_embedder.linear1.")
        .with_key_remap(r"time_embedder.mlp.2\.", r"time_embedder.linear2.")
        .with_key_remap(r"\.blocks\.", ".layers.")
        .with_key_remap(r"\.mlp\.", ".swiglu.")
        .with_key_remap(r"\.adaLN_modulation.1\.", ".ada_ln_modulation.")
        .with_key_remap(r"latent2atom_proj.0\.", "latent_to_atom_w1.")
        .with_key_remap(r"latent2atom_proj.2\.", "latent_to_atom_norm.")
        .with_key_remap(r"latent2atom_proj.3\.", "latent_to_atom_w2.")
        .with_key_remap(r"atom_feat_proj.0\.", "atom_feat_proj_lin.")
        .with_key_remap(r"atom_feat_proj.1\.", "atom_feat_proj_norm.")
        .with_key_remap(r"_proj.0\.", "_lin.")
        .with_key_remap(r"_proj.1\.", "_norm.")
        .with_key_remap(r"length_embedder.0\.", "length_embedder_lin.")
        .with_key_remap(r"length_embedder.1\.", "length_embedder_norm.")
        .with_key_remap(r"esm_s_proj.proj.0\.", "esm_s_proj.linear.")
        .with_key_remap(r"esm_s_proj.proj.1\.", "esm_s_proj.norm.")
        .with_key_remap(r"context2atom", "context_to_atom")
        .with_key_remap(r"atom2latent", "atom_to_latent")
        .with_key_remap(r"\.scale", ".gamma");

    let record: FoldingDiTRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, device)
        .expect("Should decode state successfully");

    return record;
}

pub fn get_model_record<B: Backend>(
    model_name: &str,
    cache_path: &str,
    device: &B::Device,
) -> FoldingDiTRecord<B> {
    let model_list = [
        "simplefold_100M",
        "simplefold_360M",
        "simplefold_700M",
        "simplefold_1.1B",
        "simplefold_1.6B",
        "simplefold_3B",
    ];
    let parent_url = "https://ml-site.cdn-apple.com/models/simplefold";
    assert!(
        model_list.iter().any(|e| e == &model_name),
        "model_name needs to be in {:#?}",
        model_list
    );

    let full_path = format!("{}/{}", &cache_path, &"weights");
    let ckpt_name = format!("{}.ckpt", &model_name);
    let ckpt_path = format!("{}/{}", &full_path, &ckpt_name);
    if !Path::new(&ckpt_path).is_file() {
        if !Path::new(&full_path).is_dir() {
            fs::create_dir(full_path.clone()).unwrap();
        }
        println!(
            "Weights of {} model are not downloaded \n Starting download",
            model_name
        );
        let weight_url = format!("{}/{}", &parent_url, &ckpt_name);
        match linear_download(&weight_url, &ckpt_path) {
            Ok(()) => {
                println!("Finished download successfully");
            }
            Err(e) => {
                eprintln!("Failed to download {weight_url} because {e}");
            }
        };
    }

    let record = load_weights(ckpt_path, device);
    return record;
}
