use hydrochess_wasm::{Variant, evaluation, game::GameState};
use std::time::Instant;

#[inline]
fn eval_helper(game: &GameState) -> i32 {
    #[cfg(feature = "nnue")]
    return evaluation::evaluate(game, None);
    #[cfg(not(feature = "nnue"))]
    return evaluation::evaluate(game);
}

#[test]
fn benchmark_static_eval_perf() {
    let mut game = GameState::default();
    game.setup_variant(Variant::Classical);

    let score = eval_helper(&game);
    println!("Score for default position: {}", score);

    println!("Warming up...");
    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(eval_helper(std::hint::black_box(&game)));
    }

    let iterations = 1_000_000;
    println!("Running {} iterations...", iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(eval_helper(std::hint::black_box(&game)));
    }
    let duration = start.elapsed();

    let total_ns = duration.as_nanos();
    let avg_ns = total_ns as f64 / iterations as f64;

    let time_per_eval_str = if avg_ns < 1000.0 {
        format!("{:.2} ns", avg_ns)
    } else if avg_ns < 1_000_000.0 {
        format!("{:.2} µs", avg_ns / 1000.0)
    } else {
        format!("{:.2} ms", avg_ns / 1_000_000.0)
    };

    println!("\n========================================");
    println!("Static Eval Performance Benchmark");
    println!("========================================");
    println!("Total Evals:   {}", iterations);
    println!("Total Time:    {:?}", duration);
    println!("Avg Time/Eval: {}", time_per_eval_str);
    println!(
        "Evals/Sec:     {:.2} M/s",
        (iterations as f64 / duration.as_secs_f64()) / 1_000_000.0
    );
    println!("========================================\n");
}
