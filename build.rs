use std::env;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    if target.contains("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=8388608");
    }

    // Embed git commit info so every binary self-reports which snapshot it was built from.
    // The values are empty strings when git is unavailable or the repo has no commits.
    let commit = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let date = Command::new("git")
        .args(["log", "-1", "--format=%cs", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=SPRT_GIT_COMMIT={}", commit);
    println!("cargo:rustc-env=SPRT_GIT_DATE={}", date);
    // Rebuild when the checked-out commit changes.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
}
