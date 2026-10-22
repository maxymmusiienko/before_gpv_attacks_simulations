use rand::distributions::Distribution;
use std::fs::File;
use std::io::Write;
use nalgebra::{Matrix2, Vector2};
use rand::distributions::WeightedIndex;
use rand::Rng;
use serde::Serialize;

#[derive(Serialize)]
struct SimulationOutput {
    basis_sk: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
}

struct Ggh2D {
    secret_basis: Matrix2<f64>,
    inv_secret_basis: Matrix2<f64>,
    b_tilde: Matrix2<f64>,
}

impl Ggh2D {
    fn new(secret_basis: Matrix2<f64>) -> Ggh2D {
        let inv_secret_basis = secret_basis.try_inverse()
            .expect("Unable to invert the secret basis");

        let b1: Vector2<f64> = secret_basis.column(0).into();
        let b2: Vector2<f64> = secret_basis.column(1).into();

        let b1_tilde = b1;
        let mu_21 = b2.dot(&b1_tilde) / b1_tilde.norm_squared();
        let b2_tilde = b2 - b1_tilde * mu_21;

        let b_tilde = Matrix2::from_columns(&[b1_tilde, b2_tilde]);
        Self {secret_basis, inv_secret_basis, b_tilde}
    }

    fn sign_error(&self, target_c: &Vector2<f64>) -> Vector2<f64> {
        let alpha = self.inv_secret_basis * target_c;

        let alpha_rounded = Vector2::new(alpha.x.round(), alpha.y.round());
        let lattice_point_v = self.secret_basis * alpha_rounded;

        target_c - lattice_point_v
    }

    pub fn sign_klein<R: Rng>(&self, target_c: &Vector2<f64>, s: f64, rng: &mut R) -> Vector2<f64> {
        let b1: Vector2<f64> = self.secret_basis.column(0).into();
        let b2: Vector2<f64> = self.secret_basis.column(1).into();

        let b1_tilde: Vector2<f64> = self.b_tilde.column(0).into();
        let b2_tilde: Vector2<f64> = self.b_tilde.column(1).into();

        let mut current_c = *target_c;
        let c_prime_2 = current_c.dot(&b2_tilde) / b2_tilde.norm_squared();
        let sigma_2 = s / b2_tilde.norm();
        let z2 = Ggh2D::sample_discrete_gaussian_1d(c_prime_2, sigma_2, rng);

        current_c -= b2 * (z2 as f64);

        let c_prime_1 = current_c.dot(&b1_tilde) / b1_tilde.norm_squared();
        let sigma_1 = s / b1_tilde.norm();
        let z1 = Ggh2D::sample_discrete_gaussian_1d(c_prime_1, sigma_1, rng);

        let lattice_point_v = b1 * (z1 as f64) + b2 * (z2 as f64);

        target_c - lattice_point_v
    }

    fn gso_2d(&self) -> Matrix2<f64> {
        let b1: Vector2<f64> = self.secret_basis.column(0).into();
        let b2: Vector2<f64> = self.secret_basis.column(1).into();

        let b1_tilde = b1;
        let mu_21 = b2.dot(&b1_tilde) / b1_tilde.norm_squared();
        let b2_tilde = b2 - b1_tilde * mu_21;
        Matrix2::from_columns(&[b1_tilde, b2_tilde])
    }

    fn sample_discrete_gaussian_1d(c_prime: f64, sigma: f64, rng: &mut impl Rng) -> i64 {
        let tail_cut = 4.0 * sigma;
        let z_min = (c_prime - tail_cut).floor() as i64;
        let z_max = (c_prime + tail_cut).ceil() as i64;

        let mut weights = Vec::with_capacity((z_max - z_min + 1) as usize);
        let pi = std::f64::consts::PI;
        let sigma_sq = sigma * sigma;

        for z in z_min..=z_max {
            let diff = z as f64 - c_prime;
            let weight = (-pi * diff * diff / sigma_sq).exp();
            weights.push(weight);
        }

        let dist = WeightedIndex::new(&weights)
            .expect("List must not be empty and weights must be positive");

        let chosen_idx = dist.sample(rng);
        z_min + (chosen_idx as i64)
    }
}

pub fn run_ggh_leakpk_simulation(discrete_flag : bool) {
    let mut rng = rand::thread_rng();

    let b_sk = Matrix2::new(
        3.0, 1.2,
        0.8, 2.5,
    );

    let ggh = Ggh2D::new(b_sk);
    let n_signatures = 5000;
    let mut error_points = Vec::with_capacity(n_signatures);


    let norm_b1_tilde = ggh.b_tilde.column(0).norm();
    let norm_b2_tilde = ggh.b_tilde.column(1).norm();
    let max_tilde_norm = norm_b1_tilde.max(norm_b2_tilde);

    let s = 10.0 * max_tilde_norm;

    for _ in 0..n_signatures {
        let c = Vector2::new(
            rng.gen_range(-50.0..50.0),
            rng.gen_range(-50.0..50.0),
        );
        if discrete_flag {
            let err = ggh.sign_klein(&c, s, &mut rng);
            error_points.push(vec![err.x, err.y]);
        }
        else {
            let err = ggh.sign_error(&c);
            error_points.push(vec![err.x, err.y]);
        }
    }

    let output = SimulationOutput {
        basis_sk: vec![
            vec![b_sk[(0, 0)], b_sk[(1, 0)]],
            vec![b_sk[(0, 1)], b_sk[(1, 1)]],
        ],
        errors: error_points,
    };

    let json_data = serde_json::to_string_pretty(&output).unwrap();
    let mut file = File::create("ggh_errors.json").expect("Could not create file");
    file.write_all(json_data.as_bytes()).unwrap();

    println!("Generated {n_signatures} signatures and saved to 'ggh_errors.json'");
}
