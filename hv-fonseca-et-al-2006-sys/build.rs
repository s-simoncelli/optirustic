extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let versions: [String; 2] = ["hv-1.3-src".to_string(), "hv-2.0rc2-src".to_string()];

    let lib_path = PathBuf::from("vendor")
        .join(&versions[1])
        .canonicalize()
        .expect("Cannot canonicalize path");

    let bindings = bindgen::Builder::default()
        .header(lib_path.join("hv.h").to_str().unwrap().to_string())
        .generate()
        .expect("Unable to generate 'hv' bindings");

    let file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(file)
        .expect("Couldn't write 'hv' bindings");

    // Compile library with version #4 - see section IV of the paper
    cc::Build::new()
        .define("VARIANT", "4")
        .file(lib_path.join("hv.c"))
        .compile("hv-fonseca-et-al-2006-sys");
}
