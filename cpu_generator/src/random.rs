use fasthash::murmur3::hash32;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct ParetoDistribution {
    pub x: f32,
    pub alpha: f32,
}

impl ParetoDistribution {
    pub fn new(x: f32, alpha: f32) -> Self {
        ParetoDistribution { x, alpha }
    }
}

#[inline]
pub fn uniform_to_pareto(u: f32, dist: &ParetoDistribution) -> f32 {
    dist.x / ((1.0f32 - u).powf(1.0f32 / dist.alpha))
}

#[inline]
pub fn random_property(i: u64, seed: u64) -> f32 {
    let h = hash32([i.to_be_bytes(), seed.to_be_bytes()].concat());
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    v as f32
}

#[inline]
pub fn random_edge(i: u64, j: u64, seed: u64) -> f32 {
    let h = hash32([i.to_be_bytes(), j.to_be_bytes(), seed.to_be_bytes()].concat());
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    v as f32
}

pub fn generate_seeds<const N: usize>() -> [u64; N] {
    let mut rng = rand::thread_rng();
    let mut seeds = [0u64; N];

    for i in 0..N {
        loop {
            let r: u64 = rng.gen();
            if (i == 0) || (!(seeds[0..(i-1)].contains(&r))) {
                seeds[i] = r;
                break;
            }
        }
    }

    seeds
}
