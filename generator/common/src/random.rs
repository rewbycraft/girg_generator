//! Re-export of [`generator_core::random`] + function to generate seed values.

pub use generator_core::random::*;

/// Generate seed values.
///
/// Generates `n` random [`u64`] values such that no two are the same.
/// This is important to ensure we don't have two properties that always match.
///
/// # Arguments
/// * `n` - The number of values to generate.
pub fn generate_seeds(n: usize) -> Vec<u64> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut seeds = vec![];
    seeds.resize(n, 0);

    for i in 0..n {
        loop {
            let r: u64 = rng.gen();
            if (i == 0) || (!(seeds[0..(i - 1)].contains(&r))) {
                seeds[i] = r;
                break;
            }
        }
    }

    seeds
}
