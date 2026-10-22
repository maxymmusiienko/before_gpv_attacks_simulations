use nalgebra::{Matrix2, Vector2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::io::Write;

#[derive(Serialize)]
pub struct SimulationOutput {
    pub basis_sk: Vec<Vec<f64>>,
    pub errors: Vec<Vec<f64>>,
}

/// Result of the Gram-Schmidt ortogonalization
pub struct Gso2D {
    pub b1_tilde: Vector2<f64>,
    pub b2_tilde: Vector2<f64>,
    pub b1_norm_sq: f64,
    pub b2_norm_sq: f64,
}

impl Gso2D {
    pub fn compute(b1: &Vector2<f64>, b2: &Vector2<f64>) -> Self {
        let b1_tilde = *b1;
        let b1_norm_sq = b1_tilde.norm_squared();

        let mu_21 = b2.dot(&b1_tilde) / b1_norm_sq;
        let b2_tilde = b2 - b1_tilde * mu_21;
        let b2_norm_sq = b2_tilde.norm_squared();

        Self {
            b1_tilde,
            b2_tilde,
            b1_norm_sq,
            b2_norm_sq,
        }
    }

    pub fn max_norm(&self) -> f64 {
        self.b1_norm_sq.sqrt().max(self.b2_norm_sq.sqrt())
    }
}

pub struct Ggh2D {
    pub b1: Vector2<f64>,
    pub b2: Vector2<f64>,
    pub secret_basis: Matrix2<f64>,
    pub inv_secret_basis: Matrix2<f64>,
    pub gso: Gso2D,
}

impl Ggh2D {
    pub fn new(secret_basis: Matrix2<f64>) -> Self {
        let inv_secret_basis = secret_basis
            .try_inverse()
            .expect("Secret basis must be invertible");

        let b1 = secret_basis.column(0).into();
        let b2 = secret_basis.column(1).into();
        let gso = Gso2D::compute(&b1, &b2);

        Self {
            b1,
            b2,
            secret_basis,
            inv_secret_basis,
            gso,
        }
    }

    /// Signing with Babai alg, weak
    pub fn sign_babai(&self, target_c: &Vector2<f64>) -> Vector2<f64> {
        let alpha = self.inv_secret_basis * target_c;
        let alpha_rounded = Vector2::new(alpha.x.round(), alpha.y.round());
        let lattice_point_v = self.secret_basis * alpha_rounded;

        target_c - lattice_point_v
    }

    /// Signing with Klein | GPV, strong
    pub fn sign_klein(&self, target_c: &Vector2<f64>, s: f64, rng: &mut impl Rng) -> Vector2<f64> {
        let mut current_c = *target_c;

        let c_prime_2 = current_c.dot(&self.gso.b2_tilde) / self.gso.b2_norm_sq;
        let sigma_2 = s / self.gso.b2_norm_sq.sqrt();
        let z2 = sample_discrete_gaussian_1d(c_prime_2, sigma_2, rng);

        current_c -= self.b2 * (z2 as f64);

        let c_prime_1 = current_c.dot(&self.gso.b1_tilde) / self.gso.b1_norm_sq;
        let sigma_1 = s / self.gso.b1_norm_sq.sqrt();
        let z1 = sample_discrete_gaussian_1d(c_prime_1, sigma_1, rng);

        let lattice_point_v = self.b1 * (z1 as f64) + self.b2 * (z2 as f64);

        target_c - lattice_point_v
    }
}

/// Sampling on Z
fn sample_discrete_gaussian_1d(c_prime: f64, sigma: f64, rng: &mut impl Rng) -> i64 {
    let tail_cut = 4.0 * sigma;
    let z_min = (c_prime - tail_cut).floor() as i64;
    let z_max = (c_prime + tail_cut).ceil() as i64;

    let pi = std::f64::consts::PI;
    let sigma_sq = sigma * sigma;

    let weights: Vec<f64> = (z_min..=z_max)
        .map(|z| {
            let diff = z as f64 - c_prime;
            (-pi * diff * diff / sigma_sq).exp()
        })
        .collect();

    let dist = WeightedIndex::new(&weights).expect("Failed to initialize WeightedIndex");
    z_min + (dist.sample(rng) as i64)
}

/// Simulation function
pub fn run_ggh_leakpk_simulation(discrete_flag: bool) {
    let mut rng = rand::thread_rng();

    let b_sk = Matrix2::new(
        3.0, 1.2,
        0.8, 2.5,
    );

    let ggh = Ggh2D::new(b_sk);
    let s = 1.8 * ggh.gso.max_norm(); // 1.8x достатньо для сгладжування

    let n_signatures = 5000;
    let mut error_points = Vec::with_capacity(n_signatures);

    for _ in 0..n_signatures {
        let c = Vector2::new(
            rng.gen_range(-50.0..50.0),
            rng.gen_range(-50.0..50.0),
        );

        let err = if discrete_flag {
            ggh.sign_klein(&c, s, &mut rng)
        } else {
            ggh.sign_babai(&c)
        };

        error_points.push(vec![err.x, err.y]);
    }

    let output = SimulationOutput {
        basis_sk: vec![
            vec![ggh.b1.x, ggh.b1.y],
            vec![ggh.b2.x, ggh.b2.y],
        ],
        errors: error_points,
    };

    let json_data = serde_json::to_string_pretty(&output).unwrap();
    let mut file = File::create("ggh_errors.json").expect("Could not create file");
    file.write_all(json_data.as_bytes()).unwrap();

    println!("Simulation finished. Points written to ggh_errors.json");
}
