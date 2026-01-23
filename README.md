# SimpleFold-burn

This repo contains an implementation of the simplefold model in Rust with the [burn](https://burn.dev/) library. 
The original code is provided by apple [here](https://github.com/apple/ml-simplefold/tree/main). 

>[!NOTE]
>This is a personal project used to simultaneously learn more about Rust and get up to date in the AI for drug discovery game. It is still very rough around the edges, but they will get smoother in the coming months. 
Any suggestions are welcomed.
 

## Installation 

```
git clone https://github.com/gagho/simplefold-burn
cd simplefold-burn
```

The model uses the pre-trained protein LM model [ESM](https://github.com/facebookresearch/esm)  to encode amino-acid sequences. Instead of implementing it from scratch in Rust we call the pytorch version using [pyO3](https://github.com/PyO3/pyo3) bindings.  

```
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
deactivate
```

To use the code for now you would need to tweek the main.rs file to your liking and run: 
```
cargo run --release
```

## A todo list 

- [ ] Clean up:
	- [ ] Test ESM install
	- [ ] Save configs + load any version 
	- [ ] Test all backends 
	- [ ] Copy burn [examples](https://github.com/tracel-ai/burn/tree/main/examples) structure
	- [ ] Clean up python section 
	- [ ] Deal with compile warnings 
	- [ ] Lib it
- [ ] Improve:
	- [ ] Implement batching logic
	- [ ] Read-Write CIF 
	- [ ] Add plddt and TM metrics 
	- [ ] Attention mask for atom encoder/decoder
	- [ ] Add training 
	
## Citation 

The simplefold model is described in the following paper:
```
@article{simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and Susskind, Josh and Bautista, Miguel Angel},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```
## License


This implementation is licensed under the MIT License, see [LICENSE](LICENSE) for details.

For the pre-trained weights' licenses, please refer to their original source: [Simplefold](https://github.com/apple/ml-simplefold/tree/main)
