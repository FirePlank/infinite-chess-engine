use std::collections::HashMap;
use std::fs;

fn main() {
    let json_path = "spsa_best_params.json";
    let params_path = "src/search/params.rs";

    // 1. Read the best params from JSON
    let json_content = match fs::read_to_string(json_path) {
        Ok(c) => c,
        Err(_) => {
            eprintln!(
                "Error: Cannot find {}. Run the tuner first to generate best params.",
                json_path
            );
            std::process::exit(1);
        }
    };

    let best_params: HashMap<String, serde_json::Value> =
        serde_json::from_str(&json_content).expect("Failed to parse JSON");

    // 2. Read the params.rs file
    let mut params_content = fs::read_to_string(params_path).expect("Failed to read params.rs");

    println!("Applying tuned parameters to constants...");

    let mut update_count = 0;

    for (name, val) in best_params {
        // Construct the constant name, e.g., razoring_linear -> DEFAULT_RAZORING_LINEAR
        let const_name = format!("DEFAULT_{}", name.to_uppercase());

        // Get the integer value as string
        let new_value_str = if let Some(f) = val.as_f64() {
            f.round().to_string()
        } else if let Some(i) = val.as_i64() {
            i.to_string()
        } else {
            continue;
        };

        // We process the file line by line to find and replace the constant value
        let lines: Vec<String> = params_content.lines().map(|s| s.to_string()).collect();
        let mut new_lines = Vec::new();
        let mut found_limit = false;

        for line in lines {
            // Match pattern: pub const DEFAULT_NAME: type = value;
            if line.trim().starts_with("pub const ")
                && line.contains(&const_name)
                && let Some(eq_idx) = line.find('=')
                && let Some(semi_idx) = line.find(';')
                && eq_idx < semi_idx
            {
                let prefix = &line[..eq_idx + 1];
                let suffix = &line[semi_idx..];
                // Reconstruct line with new value
                let new_line = format!("{} {}{}", prefix, new_value_str, suffix);
                new_lines.push(new_line);
                found_limit = true;
                continue;
            }
            new_lines.push(line);
        }

        if found_limit {
            params_content = new_lines.join("\n");
            // Maintain trailing newline
            params_content.push('\n');
            update_count += 1;
        } else {
            println!(
                "  [!] Warning: Constant {} not found in params.rs",
                const_name
            );
        }
    }

    // 3. Write back the updated file
    fs::write(params_path, params_content).expect("Failed to write updated params.rs");

    println!(
        "Successfully updated {} constants in {}",
        update_count, params_path
    );
}
