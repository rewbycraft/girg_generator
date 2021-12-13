use tracing::info;

pub mod random;
pub mod threads;

#[derive(Clone, Debug)]
pub struct GenerationParameters {
    pub seeds: [u64; 4],
    pub pareto: random::ParetoDistribution,
    pub alpha: f32,
    pub w: f32,
    pub v: u64,
}

impl GenerationParameters {
    pub fn new(pareto: random::ParetoDistribution, alpha: f32, v: u64) -> Self {
        let seeds = random::generate_seeds();

        Self::from_seeds(pareto, alpha, v, seeds)
    }

    pub fn from_seeds(pareto: random::ParetoDistribution, alpha: f32, v: u64, seeds: [u64; 4]) -> Self {
        let mut s = Self {
            seeds,
            pareto,
            alpha,
            w: 0.0,
            v,
        };

        s.compute_w();
        s
    }

    pub fn compute_w(&mut self) {
        info!("Computing W...");
        self.w = (0..self.v)
            .map(|i| random::uniform_to_pareto(
                random::random_property(i, self.seeds[0]),
                &self.pareto)
            ).sum();
        info!("Computed W = {}", self.w);
    }
}

#[inline]
pub fn compute_distance(p0_i: f32, p1_i: f32, p0_j: f32, p1_j: f32) -> f32 {
    fn dist_c(i: f32, j: f32) -> f32 {
        (i - j).abs().min(1.0f32 - ((i - j).abs()))
    }
    dist_c(p0_i, p0_j).max(dist_c(p1_i, p1_j))
}

#[inline]
pub fn compute_probability(d: f32, w_i: f32, w_j: f32, params: &GenerationParameters) -> f32 {
    if params.alpha.is_infinite() {
        let v = ((w_i * w_j) / params.w).powf(1.0f32 / 2.0f32);
        if d <= v {
            1.0f32
        } else {
            0.0f32
        }
    } else {
        ((((w_i * w_j) / params.w).powf(params.alpha)) / (d.powf(params.alpha * 2.0f32))).min(1.0f32)
    }
}

#[inline]
pub fn generate_edge(i: u64, j: u64, w_i: f32, p0_i: f32, p1_i: f32, w_j: f32, p0_j: f32, p1_j: f32, params: &GenerationParameters) -> bool {
    let d = compute_distance(p0_i, p1_i, p0_j, p1_j);
    let p = compute_probability(d, w_i, w_j, params);
    p < random::random_edge(i, j, params.seeds[1])
}

#[inline]
pub fn worker_function<F: FnMut(u64, u64)>(start: (u64, u64), end: (u64, u64), params: &GenerationParameters, mut cb: F) {
    todo!("fix bugs");
    let mut i = start.0;
    let mut j = start.1;

    // Pre-calculate the params for j.
    let mut w_j = random::uniform_to_pareto(random::random_property(j, params.seeds[0]), &params.pareto);
    let mut p0_j = random::random_property(j, params.seeds[2]);
    let mut p1_j = random::random_property(j, params.seeds[3]);

    loop {
        let w_i = random::uniform_to_pareto(random::random_property(i, params.seeds[0]), &params.pareto);
        let p0_i = random::random_property(i, params.seeds[2]);
        let p1_i = random::random_property(i, params.seeds[3]);

        if generate_edge(i, j, w_i, p0_i, p1_i, w_j, p0_j, p1_j, params) {
            cb(i, j)
        }

        // Increment i,j
        i += 1;
        if i >= params.v {
            i = 0;
            j += 1;

            // Re-calculate the params for j.
            w_j = random::uniform_to_pareto(random::random_property(j, params.seeds[0]), &params.pareto);
            p0_j = random::random_property(j, params.seeds[2]);
            p1_j = random::random_property(j, params.seeds[3]);
        }
        if j >= params.v {
            // We've past the last node.
            break;
        }
        if i >= end.0 && j >= end.1 {
            // We're done.
            break;
        }
    }
}

#[inline]
pub fn worker_function_pregen<F: FnMut(u64, u64)>(start: (u64, u64), end: (u64, u64), params: &GenerationParameters, mut cb: F) {
    let mut i = start.0;
    let mut j = start.1;

    // Pre-calculate the params for j.
    let ws: Vec<f32> = (0..params.v).map(|j| random::uniform_to_pareto(random::random_property(j, params.seeds[0]), &params.pareto)).collect();
    let p0s: Vec<f32> = (0..params.v).map(|j| random::random_property(j, params.seeds[2])).collect();
    let p1s: Vec<f32> = (0..params.v).map(|j| random::random_property(j, params.seeds[3])).collect();

    loop {
        if generate_edge(i,
                         j,
                         *ws.get(i as usize).unwrap(),
                         *p0s.get(i as usize).unwrap(),
                         *p1s.get(i as usize).unwrap(),
                         *ws.get(j as usize).unwrap(),
                         *p0s.get(j as usize).unwrap(),
                         *p1s.get(j as usize).unwrap(),
                         params)
        {
            cb(i, j)
        }

        // Increment i,j
        i += 1;
        if i >= params.v.min(end.0) {
            i = start.0;
            j += 1;
        }
        if j >= params.v.min(end.1) {
            // We've past the last node.
            break;
        }
        if i >= end.0 && j >= end.1 {
            // We're done.
            break;
        }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
