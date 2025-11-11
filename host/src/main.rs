use anyhow::{anyhow, Result};
use clap::Parser;
use methods::{METHODS_ELF, METHODS_ID};
use serde::{Deserialize, Serialize};

// Types partagés (host <-> guest)
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

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use risc0_zkvm::{default_prover, ExecutorEnv};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "ZK Linear Regression (RISC Zero)")]
struct Args {
    #[arg(long)]
    dataset: Option<String>,
    #[arg(long)]
    synthetic: bool,
    #[arg(long, default_value_t = 150)]
    n: usize,
    #[arg(long, default_value_t = 4)]
    d: usize,
    #[arg(long, default_value_t = 0.05)]
    noise: f64,
    #[arg(long, default_value_t = 10000.0)]
    epsilon: f64,
    #[arg(long, default_value_t = 100)]
    scale: i64,
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.dataset.is_none() && !args.synthetic {
        return Err(anyhow!("Specify --dataset path or --synthetic"));
    }

    println!("=== ZK Linear Regression on Iris ===\n");

    let (x_real, y_real, feature_names) = if let Some(path) = args.dataset.clone() {
        load_iris(&path)?
    } else {
        gen_synth(args.n, args.d)?
    };

    let (n, d) = (x_real.nrows(), x_real.ncols());
    
    // Show input data preview
    println!("Input Data:");
    if args.verbose {
        println!("  Dataset: n={} d={} features={:?}", n, d, feature_names);
        println!("  X (first 5 rows): {:?}...", &x_real.row(0).iter().take(d.min(5)).copied().collect::<Vec<_>>());
        println!("  Y (first 10): {:?}...", &y_real.iter().take(10).copied().collect::<Vec<_>>());
    } else {
        println!("  Samples: {}, Features: {}", n, d);
    }
    println!();

    // Train w (least squares)
    let w_real = {
        let xtx = &x_real.transpose() * &x_real;
        let xty = &x_real.transpose() * &y_real;
        xtx.lu().solve(&xty).ok_or_else(|| anyhow!("Matrix inversion failed"))?
    };

    // Scale to fixed-point
    let scale_f = args.scale as f64;
    let x_scaled: Vec<i64> = x_real.iter().map(|v| (v * scale_f).round() as i64).collect();
    let y_scaled: Vec<i64> = y_real.iter().map(|v| (v * scale_f).round() as i64).collect();
    let w_scaled: Vec<i64> = w_real.iter().map(|v| (v * scale_f).round() as i64).collect();

    // Hash data
    let mut hasher = Sha256::new();
    for val in x_real.iter() {
        hasher.update(val.to_le_bytes());
    }
    for val in y_real.iter() {
        hasher.update(val.to_le_bytes());
    }
    let hash_data: [u8; 32] = hasher.finalize().into();

    let epsilon_scaled = (args.epsilon * scale_f).ceil() as i64;

    let input = LinRegInput {
        n,
        d,
        scale: args.scale,
        epsilon_scaled,
        x: x_scaled,
        y: y_scaled,
        w: w_scaled.clone(),
        hash_data,
    };

    // Build execution environment
    let env = ExecutorEnv::builder().write(&input)?.build()?;

    // === PROVING PHASE ===
    println!("🔒 Starting proving phase...");
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, METHODS_ELF)?;
    let prove_time = start.elapsed();
    println!("✅ Proof generated in {:.9}s", prove_time.as_secs_f64());

    // Decode output
    let output: LinRegOutput = prove_info.receipt.journal.decode()?;
    
    // Display results
    println!("\n📊 Linear Regression Results:");
    println!("  Coefficients (scaled):");
    for (i, &w_val) in output.w.iter().enumerate() {
        let w_real = w_val as f64 / scale_f;
        if i < feature_names.len() {
            println!("    {}: {:.6} (scaled: {})", feature_names[i], w_real, w_val);
        } else {
            println!("    w[{}]: {:.6} (scaled: {})", i, w_real, w_val);
        }
    }
    let mse = (output.mse_num as f64) / (output.mse_den as f64);
    println!("  MSE:            {:.6}", mse);
    println!("  R² Score:       {:.6}", 1.0 - mse / variance(&y_real));
    
    // === VERIFICATION PHASE ===
    println!("\n🔍 Starting verification phase...");
    let start_v = Instant::now();
    prove_info.receipt.verify(METHODS_ID)?;
    let verify_time = start_v.elapsed();
    println!("✅ Proof verified in {:.6}ms", verify_time.as_secs_f64() * 1000.0);

    // === PERFORMANCE ANALYSIS ===
    println!("\n📈 Performance Analysis:");
    println!("  Proof Size:      {} bytes", prove_info.receipt.journal.bytes.len());
    println!("  Proving Time:    {:.9}s", prove_time.as_secs_f64());
    println!("  Verification:    {:.6}ms", verify_time.as_secs_f64() * 1000.0);
    println!("  Total Time:      {:.9}s", (prove_time + verify_time).as_secs_f64());
    
    // === CORRECTNESS CHECK ===
    println!("\n✓ Correctness Check:");
    println!("  Max Gradient:    {} (threshold: {})", output.max_grad_scaled, epsilon_scaled);
    println!("  Data Hash:       0x{}", hex::encode(&output.hash_data[..8]));
    if output.max_grad_scaled <= epsilon_scaled {
        println!("  ✅ Gradient check passed!");
    } else {
        println!("  ⚠️  Gradient exceeds threshold");
    }
    
    // Match input/output hashes
    if output.hash_data == hash_data {
        println!("  ✅ Data integrity verified!");
    } else {
        println!("  ❌ Data hash mismatch!");
    }

    Ok(())
}

fn variance(y: &DVector<f64>) -> f64 {
    let mean = y.mean();
    y.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / y.len() as f64
}

fn gen_synth(n: usize, d: usize) -> Result<(DMatrix<f64>, DVector<f64>, Vec<String>)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut x = DMatrix::<f64>::zeros(n, d);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 1..d {
            x[(i, j)] = rng.random_range(-1.0..1.0);
        }
    }
    let true_w = DVector::<f64>::from_fn(d, |i, _| {
        if i == 0 { 0.5 } else { rng.random_range(-2.0..2.0) }
    });
    let mut y = DVector::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..d {
            acc += x[(i, j)] * true_w[j];
        }
        let noise: f64 = rng.random_range(-0.01..0.01);
        y[i] = acc + noise;
    }
    let names = (0..d).map(|j| format!("f{}", j)).collect();
    Ok((x, y, names))
}

fn load_iris(path: &str) -> Result<(DMatrix<f64>, DVector<f64>, Vec<String>)> {
    let file = File::open(Path::new(path))?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(BufReader::new(file));

    let mut features: Vec<f64> = Vec::new();
    let mut target: Vec<f64> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() < 4 { continue; }
        let sepal_length: f64 = rec[0].parse()?;
        let sepal_width: f64 = rec[1].parse()?;
        let petal_length: f64 = rec[2].parse()?;
        let petal_width: f64 = rec[3].parse()?;
        target.push(petal_length);
        features.push(1.0); // bias
        features.push(sepal_length);
        features.push(sepal_width);
        features.push(petal_width);
    }
    let n = target.len();
    let d = 4;
    let x = DMatrix::from_row_slice(n, d, &features);
    let y = DVector::from_row_slice(&target);
    let names = vec!["bias".into(), "sepal_length".into(), "sepal_width".into(), "petal_width".into()];
    Ok((x, y, names))
}