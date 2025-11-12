use anyhow::{anyhow, Result};
use clap::Parser;
use methods::{METHODS_ELF, METHODS_ID};
use serde::{Deserialize, Serialize};
use plotters::prelude::*;

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

    // 1. Plot the linear regression
    plot_linear_regression(
        &x_real,
        &y_real,
        &output.w,
        args.scale,
        "linear_regression_plot.png",
    )?;

    // 2. Plot the residuals
    {
        // Convertir les poids (w) en f64
        let w_real_vec: Vec<f64> = output.w.iter().map(|&wi| wi as f64 / scale_f).collect();
        let w_real = DVector::from_vec(w_real_vec);

        // Calculer les valeurs prédites: y_pred = X * w
        let y_pred = &x_real * &w_real;

        // Calculer les résidus: residuals = y_real - y_pred
        let residuals_vec: Vec<f64> = (&y_real - y_pred).iter().cloned().collect();

        let root_area = BitMapBackend::new("residuals_plot.png", (640, 480)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let min_res = residuals_vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_res = residuals_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root_area)
            .caption("Residuals", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0i32..(residuals_vec.len() as i32), min_res..max_res)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(residuals_vec.iter().enumerate().map(|(i, &res_val)| {
            Rectangle::new([(i as i32, 0.0), (i as i32 + 1, res_val)], BLUE.filled())
        }))?;

        root_area.present()?;
        println!("✅ Residuals plot saved to residuals_plot.png");
    };

    // 3. Plot the regression coefficients
    {
        let root_area = BitMapBackend::new("coefficients_plot.png", (640, 480)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let w_real: Vec<f64> = output.w.iter().map(|&wi| wi as f64 / scale_f).collect();
        let min_w = w_real.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_w = w_real.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range = (min_w * 1.2)..(max_w * 1.2);

        let mut chart = ChartBuilder::on(&root_area)
            .caption("Regression Coefficients", ("sans-serif", 40).into_font())
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            .build_cartesian_2d(feature_names.into_segmented(), y_range)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .bold_line_style(&BLACK.mix(0.2))
            .draw()?;

        chart.draw_series(
            Histogram::vertical(&chart)
                .style(BLUE.filled())
                .data(feature_names.iter().zip(w_real.iter()).map(|(name, &val)| (name, val)))
        )?;

        root_area.present()?;
        println!("✅ Coefficients plot saved to coefficients_plot.png");
    };

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
        .has_headers(true)
        .from_reader(file);

    let mut features: Vec<f64> = Vec::new();
    let mut target: Vec<f64> = Vec::new();

    // Headers: sepal_length,sepal_width,petal_length,petal_width,species
    for result in rdr.records() {
        let record = result?;
        // Add bias term
        features.push(1.0);
        // sepal_length
        features.push(record[0].parse::<f64>()?);
        // sepal_width
        features.push(record[1].parse::<f64>()?);
        // petal_width
        features.push(record[3].parse::<f64>()?);
        // Target: petal_length
        target.push(record[2].parse::<f64>()?);
    }

    let n = target.len();
    let d = 4; // bias + 3 features
    let x = DMatrix::from_row_slice(n, d, &features);
    let y = DVector::from_vec(target);

    let names = vec![
        "bias".into(),
        "sepal_length".into(),
        "sepal_width".into(),
        "petal_width".into(),
    ];
    Ok((x, y, names))
}

fn plot_linear_regression(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    w: &Vec<i64>,
    scale: i64,
    file_name: &str,
) -> Result<()> {
    let root_area = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Déterminer les plages du graphique à partir des données
    // Nous utilisons la deuxième colonne de x (par exemple, sepal_length) pour l'axe des x
    let min_x = x.column(1).min();
    let max_x = x.column(1).max();
    let min_y = y.min();
    let max_y = y.max();

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Linear Regression", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    // Dessiner les points de données (nuage de points)
    chart.draw_series(
        x.column(1)
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| Circle::new((xi, yi), 3, BLUE.filled())),
    )?
    .label("Data Points")
    .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    // Dessiner la ligne de régression
    let w_real: Vec<f64> = w.iter().map(|&wi| wi as f64 / scale as f64).collect();
    let line_series = LineSeries::new(
        (0..=100).map(|i| {
            let x_val = min_x + (max_x - min_x) * (i as f64 / 100.0);
            // Note : Ceci suppose un modèle simple y = w0 + w1*x1.
            // Pour un modèle multivarié, vous devez choisir quelles caractéristiques visualiser.
            // Ici, nous supposons que w[0] est le biais et w[1] est le coefficient pour x.column(1).
            let y_val = w_real[0] + w_real[1] * x_val;
            (x_val, y_val)
        }),
        &RED,
    );

    chart
        .draw_series(line_series)?
        .label("Regression Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root_area.present()?;
    println!("✅ Plot saved to {}", file_name);
    Ok(())
}


