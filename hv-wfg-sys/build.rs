extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let lib_path = PathBuf::from("vendor")
        .canonicalize()
        .expect("Cannot canonicalize path");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/wfg.c");

    let bindings = bindgen::Builder::default()
        .header(lib_path.join("wfg.h").to_str().unwrap().to_string())
        .generate()
        .expect("Unable to generate 'wfg' bindings");

    let file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(file)
        .expect("Couldn't write 'wfg' bindings");

    // Compile library for minimisation problems
    cc::Build::new()
        .define("MAXIMISING", "false")
        .define("opt", "2")
        .define("DEBUG", "false")
        // .flag("-Wall")
        .warnings(false)
        .file(lib_path.join("wfg.c"))
        .compile("hv-wfg-sys");
}
