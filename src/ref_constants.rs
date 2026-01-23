use std::collections::HashMap;

// pub fn tokens() -> Vec<&'static str> {
//     let toks = vec![
//         "<pad>", "-", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
//         "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", "A", "G", "C", "U",
//         "N", "DA", "DG", "DC", "DT", "DN",
//     ];
//     return toks;
// }

pub fn get_all_refs() -> (
    HashMap<String, Vec<String>>,
    Vec<&'static str>,
    HashMap<String, String>,
) {
    let mut refs: HashMap<&str, Vec<&str>> = HashMap::new();
    // refs.insert("PAD", [""]);
    refs.insert("UNK", vec!["N", "CA", "C", "O", "CB"]);
    refs.insert("PAD", vec![]);
    refs.insert("UNK", vec!["N", "CA", "C", "O", "CB"]);
    refs.insert("-", vec![]);
    refs.insert("ALA", vec!["N", "CA", "C", "O", "CB"]);
    refs.insert(
        "ARG",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
        ],
    );
    refs.insert("ASN", vec!["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"]);
    refs.insert("ASP", vec!["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]);
    refs.insert("CYS", vec!["N", "CA", "C", "O", "CB", "SG"]);
    refs.insert(
        "GLN",
        vec!["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    );
    refs.insert(
        "GLU",
        vec!["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    );
    refs.insert("GLY", vec!["N", "CA", "C", "O"]);
    refs.insert(
        "HIS",
        vec!["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    );
    refs.insert("ILE", vec!["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"]);
    refs.insert("LEU", vec!["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"]);
    refs.insert(
        "LYS",
        vec!["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    );
    refs.insert("MET", vec!["N", "CA", "C", "O", "CB", "CG", "SD", "CE"]);
    refs.insert(
        "PHE",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
        ],
    );
    refs.insert("PRO", vec!["N", "CA", "C", "O", "CB", "CG", "CD"]);
    refs.insert("SER", vec!["N", "CA", "C", "O", "CB", "OG"]);
    refs.insert("THR", vec!["N", "CA", "C", "O", "CB", "OG1", "CG2"]);
    refs.insert(
        "TRP",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ],
    );
    refs.insert(
        "TYR",
        vec![
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
        ],
    );
    refs.insert("VAL", vec!["N", "CA", "C", "O", "CB", "CG1", "CG2"]);
    refs.insert(
        "A",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9",
            "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        ],
    );
    refs.insert(
        "G",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9",
            "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        ],
    );
    refs.insert(
        "C",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1",
            "C2", "O2", "N3", "C4", "N4", "C5", "C6",
        ],
    );
    refs.insert(
        "U",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1",
            "C2", "O2", "N3", "C4", "O4", "C5", "C6",
        ],
    );
    refs.insert(
        "N",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        ],
    );
    refs.insert(
        "DA",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8",
            "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        ],
    );
    refs.insert(
        "DG",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8",
            "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        ],
    );
    refs.insert(
        "DC",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2",
            "O2", "N3", "C4", "N4", "C5", "C6",
        ],
    );
    refs.insert(
        "DT",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2",
            "O2", "N3", "C4", "O4", "C5", "C7", "C6",
        ],
    );
    refs.insert(
        "DN",
        vec![
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        ],
    );
    let mut str_refs: HashMap<String, Vec<String>> = HashMap::new();
    for (key, val) in refs.iter() {
        str_refs.insert(key.to_string(), val.iter().map(|s| s.to_string()).collect());
    }

    let tokens = vec![
        "<pad>", "-", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
        "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", "A", "G", "C", "U",
        "N", "DA", "DG", "DC", "DT", "DN",
    ];

    let mut restype_1to3: HashMap<String, String> = HashMap::new();
    restype_1to3.insert("A".to_string(), "ALA".to_string());
    restype_1to3.insert("R".to_string(), "ARG".to_string());
    restype_1to3.insert("N".to_string(), "ASN".to_string());
    restype_1to3.insert("D".to_string(), "ASP".to_string());
    restype_1to3.insert("C".to_string(), "CYS".to_string());
    restype_1to3.insert("Q".to_string(), "GLN".to_string());
    restype_1to3.insert("E".to_string(), "GLU".to_string());
    restype_1to3.insert("G".to_string(), "GLY".to_string());
    restype_1to3.insert("H".to_string(), "HIS".to_string());
    restype_1to3.insert("I".to_string(), "ILE".to_string());
    restype_1to3.insert("L".to_string(), "LEU".to_string());
    restype_1to3.insert("K".to_string(), "LYS".to_string());
    restype_1to3.insert("M".to_string(), "MET".to_string());
    restype_1to3.insert("F".to_string(), "PHE".to_string());
    restype_1to3.insert("P".to_string(), "PRO".to_string());
    restype_1to3.insert("S".to_string(), "SER".to_string());
    restype_1to3.insert("T".to_string(), "THR".to_string());
    restype_1to3.insert("W".to_string(), "TRP".to_string());
    restype_1to3.insert("Y".to_string(), "TYR".to_string());
    restype_1to3.insert("V".to_string(), "VAL".to_string());

    return (str_refs, tokens, restype_1to3);
}
