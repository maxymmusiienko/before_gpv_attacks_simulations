use std::fs::File;
use std::io::Write;
use nalgebra::{Matrix2, Vector2};
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
}

impl Ggh2D {
    fn new(secret_basis: Matrix2<f64>) -> Ggh2D {
        let inv_secret_basis = secret_basis.try_inverse()
            .expect("Unable to invert the secret basis");
        Self {secret_basis, inv_secret_basis}
    }

    fn sign_error(&self, target_c: &Vector2<f64>) -> Vector2<f64> {
        let alpha = self.inv_secret_basis * target_c;

        let alpha_rounded = Vector2::new(alpha.x.round(), alpha.y.round());
        let lattice_point_v = self.secret_basis * alpha_rounded;

        target_c - lattice_point_v
    }
}

pub fn run_ggh_leakpk_simulation() {
    let mut rng = rand::thread_rng();

    let b_sk = Matrix2::new(
        3.0, 1.2,
        0.8, 2.5,
    );

    let ggh = Ggh2D::new(b_sk);
    let n_signatures = 5000;
    let mut error_points = Vec::with_capacity(n_signatures);

    for _ in 0..n_signatures {
        let c = Vector2::new(
            rng.gen_range(-50.0..50.0),
            rng.gen_range(-50.0..50.0),
        );

        let err = ggh.sign_error(&c);
        error_points.push(vec![err.x, err.y]);
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
