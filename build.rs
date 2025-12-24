use std::env;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    if target.contains("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=8388608");
    }
}
