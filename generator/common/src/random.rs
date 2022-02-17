pub use generator_core::random::*;

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
