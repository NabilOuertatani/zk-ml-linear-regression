#![no_main]
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LinRegInput {
    pub n: usize,
    pub d: usize,
    pub scale: i64,
    pub epsilon_scaled: i64,
    pub x: Vec<i64>,
    pub y: Vec<i64>,
    pub w: Vec<i64>,
    pub hash_data: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LinRegOutput {
    pub n: usize,
    pub d: usize,
    pub scale: i64,
    pub epsilon_scaled: i64,
    pub max_grad_scaled: i64,
    pub mse_num: i128,
    pub mse_den: i128,
    pub w: Vec<i64>,
    pub hash_data: [u8; 32],
}

risc0_zkvm::guest::entry!(main);

fn main() {
    let input: LinRegInput = env::read();

    let LinRegInput {
        n,
        d,
        scale,
        epsilon_scaled,
        x,
        y,
        w,
        hash_data,
    } = input;

    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert_eq!(w.len(), d);

    // residuals r_i (scaled by scale)
    let mut residuals: Vec<i64> = vec![0; n];
    for i in 0..n {
        let mut acc: i128 = 0;
        for j in 0..d {
            let x_ij = x[i * d + j] as i128;
            let w_j = w[j] as i128;
            acc += x_ij * w_j; // scale^2 * (Xw)_real part
        }
        // divide once by scale to keep residual scaled (scale * residual_real)
        let pred_scaled = (acc / (scale as i128)) as i64;
        residuals[i] = pred_scaled - y[i];
    }

    // gradient g_j = sum_i X_ij * r_i / scale  (scaled by scale)
    let mut max_grad_abs: i64 = 0;
    for j in 0..d {
        let mut acc: i128 = 0;
        for i in 0..n {
            acc += (x[i * d + j] as i128) * (residuals[i] as i128);
        }
        let g_scaled = (acc / (scale as i128)) as i64; // scale * g_real
        let abs_g = g_scaled.abs();
        if abs_g > max_grad_abs {
            max_grad_abs = abs_g;
        }
    }

    // Check stationarity condition
    assert!(
        max_grad_abs <= epsilon_scaled,
        "Gradient infinity norm {} > epsilon {}",
        max_grad_abs,
        epsilon_scaled
    );

    // MSE: sum r_i^2 / (n * scale^2)
    let mut sum_r2: i128 = 0;
    for r in &residuals {
        let rr = *r as i128;
        sum_r2 += rr * rr;
    }
    let mse_den: i128 = (n as i128) * (scale as i128) * (scale as i128);

    let output = LinRegOutput {
        n,
        d,
        scale,
        epsilon_scaled,
        max_grad_scaled: max_grad_abs,
        mse_num: sum_r2,
        mse_den,
        w,
        hash_data,
    };

    env::commit(&output);
}