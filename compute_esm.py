import sys

sys.path.append(".venv/lib/python3.12/site-packages")
import torch

restypes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)
unk_restype_index = restype_num
restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def get_esm():
    import esm

    esm_model, esm_dict = esm.pretrained.esm2_t36_3B_UR50D()
    return esm_model, esm_dict


def af2_idx_to_esm_idx(aa, mask, af2_to_esm):
    aa = (aa + 1).masked_fill(mask != 1, 0)
    return af2_to_esm[aa]


def collate_dense_tensors(samples, pad_v=0):
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def encode_sequence(
    seq: str,
    residue_index_offset=512,
    chain_linker="G" * 25,
):
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = restype_order_with_x["X"]
    encoded = torch.tensor([restype_order_with_x.get(aa, unk_idx) for aa in seq])
    residx = torch.arange(len(encoded))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += i * residue_index_offset
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.int64)

    return encoded, residx, linker_mask, chain_index


def batch_encode_sequences(
    sequences,
    residue_index_offset=512,
    chain_linker="G" * 25,
):
    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors([aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list])
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list


def compute_language_model_representations(esmaa, esm_model, esm_dict):
    batch_size = esmaa.size(0)
    bosi, eosi = esm_dict.cls_idx, esm_dict.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), esm_dict.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # Use the first padding index as eos during inference.
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi
    res = esm_model(
        esmaa,
        repr_layers=range(esm_model.num_layers + 1),
        need_head_weights=False,
    )
    esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
    esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
    return esm_s, None


def _af2_to_esm(d):
    # Remember that t is shifted from residue_constants by 1 (0 is padding).
    esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in restypes_with_x]
    return torch.tensor(esm_reorder)


def process_esm(sequence):
    L = len(sequence)
    num_tokens = [L]
    sequence = [sequence]
    B = len(sequence)
    esm_model, esm_dict = get_esm()
    af2_to_esm = _af2_to_esm(esm_dict)
    esm_model.eval()

    aatype, mask, residx, linker_mask, _ = batch_encode_sequences(
        sequence,
        residue_index_offset=512,
        chain_linker="G" * 25,
    )

    if residx is None:
        residx = torch.arange(L).expand_as(aatype)

    esmaa = af2_idx_to_esm_idx(aatype, mask, af2_to_esm)
    esm_s_, _ = compute_language_model_representations(esmaa, esm_model, esm_dict)

    esm_s_ = esm_s_.detach()
    mask, linker_mask = mask.detach().bool(), linker_mask.detach().bool()

    esm_s = torch.zeros((B, L, esm_model.num_layers + 1, esm_s_.shape[-1]))
    true_mask = linker_mask & mask

    for i in range(B):
        true_len = true_mask[i].sum()
        assert true_len == num_tokens[i]
        esm_s[i, :true_len] = esm_s_[i, true_mask[i]]

    return esm_s[0].cpu().numpy()
