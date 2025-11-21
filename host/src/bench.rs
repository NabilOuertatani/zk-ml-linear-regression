use crate::{gen_synth, LinRegInput, LinRegOutput};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Benchmark configuration for a single test case
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub name: String,
    pub n: usize,      // Number of samples
    pub d: usize,      // Number of features
    pub scale: i64,    // Fixed-point scale
    pub epsilon: f64,  // Gradient threshold
}

/// Results from a single benchmark run
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub n: usize,
    pub d: usize,
    pub scale: i64,
    pub proving_time_ms: f64,
    pub verification_time_ms: f64,
    pub total_time_ms: f64,
    pub proof_size_bytes: usize,
    pub mse: f64,
    pub max_gradient: i64,
    pub success: bool,
}

/// Collection of all benchmark results
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
    pub timestamp: String,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            timestamp: chrono::Local::now().to_rfc3339(),
        }
    }

    /// Run a single benchmark test
    pub fn run_benchmark(
        &mut self,
        config: BenchmarkConfig,
        methods_elf: &[u8],
        methods_id: [u32; 8],
    ) -> Result<()> {
        println!("\n Running benchmark: {}", config.name);
        println!("   Samples: {}, Features: {}", config.n, config.d);

        // Generate synthetic data
        let (x_real, y_real, _) = gen_synth(config.n, config.d)?;

        // Train weights using least squares
        let w_real = {
            let xtx = &x_real.transpose() * &x_real;
            let xty = &x_real.transpose() * &y_real;
            xtx.lu()
                .solve(&xty)
                .ok_or_else(|| anyhow::anyhow!("Matrix inversion failed"))?
        };

        // Scale to fixed-point
        let scale_f = config.scale as f64;
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

        let epsilon_scaled = (config.epsilon * scale_f).ceil() as i64;

        let input = LinRegInput {
            n: config.n,
            d: config.d,
            scale: config.scale,
            epsilon_scaled,
            x: x_scaled,
            y: y_scaled,
            w: w_scaled.clone(),
            hash_data,
        };

        // Build execution environment
        let env = ExecutorEnv::builder().write(&input)?.build()?;

        // PROVING PHASE 
        let prover = default_prover();
        let prove_start = Instant::now();
        let prove_info = prover.prove(env, methods_elf)?;
        let proving_time = prove_start.elapsed();

        // Decode output
        let output: LinRegOutput = prove_info.receipt.journal.decode()?;

        //  VERIFICATION PHASE 
        let verify_start = Instant::now();
        prove_info.receipt.verify(methods_id)?;
        let verification_time = verify_start.elapsed();

        // Calculate MSE
        let mse = (output.mse_num as f64) / (output.mse_den as f64);

        // Create result
        let result = BenchmarkResult {
            name: config.name.clone(),
            n: config.n,
            d: config.d,
            scale: config.scale,
            proving_time_ms: proving_time.as_secs_f64() * 1000.0,
            verification_time_ms: verification_time.as_secs_f64() * 1000.0,
            total_time_ms: (proving_time + verification_time).as_secs_f64() * 1000.0,
            proof_size_bytes: prove_info.receipt.journal.bytes.len(),
            mse,
            max_gradient: output.max_grad_scaled,
            success: output.max_grad_scaled <= epsilon_scaled,
        };

        println!("    Proving time: {:.2}ms", result.proving_time_ms);
        println!("    Verification time: {:.2}ms", result.verification_time_ms);
        println!("    Proof size: {} bytes", result.proof_size_bytes);

        self.results.push(result);
        Ok(())
    }

    /// Save results to JSON file
    pub fn save_to_file(&self, filename: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        println!("\n Benchmark results saved to {}", filename);
        Ok(())
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        
        println!("           BENCHMARK SUMMARY");
        

        for result in &self.results {
            println!("\n {}", result.name);
            println!("   Dataset: n={}, d={}", result.n, result.d);
            println!("   Proving: {:.2}ms", result.proving_time_ms);
            println!("   Verification: {:.2}ms", result.verification_time_ms);
            println!("   Proof Size: {} bytes", result.proof_size_bytes);
            println!("   MSE: {:.6}", result.mse);
            println!("   Status: {}", if result.success { " PASS" } else { " FAIL" });
        }

        // Calculate averages
        let avg_prove = self.results.iter().map(|r| r.proving_time_ms).sum::<f64>()
            / self.results.len() as f64;
        let avg_verify = self.results.iter().map(|r| r.verification_time_ms).sum::<f64>()
            / self.results.len() as f64;
        let avg_size = self.results.iter().map(|r| r.proof_size_bytes).sum::<usize>()
            / self.results.len();

        println!("\n");
        println!("AVERAGES:");
        println!("   Proving: {:.2}ms", avg_prove);
        println!("   Verification: {:.2}ms", avg_verify);
        println!("   Proof Size: {} bytes", avg_size);
        println!("\n");
    }
}

/// Predefined benchmark configurations
pub fn get_standard_benchmarks() -> Vec<BenchmarkConfig> {
    vec![
        BenchmarkConfig {
            name: "Small Dataset (10x3)".to_string(),
            n: 10,
            d: 3,
            scale: 100,
            epsilon: 1000.0,
        },
        BenchmarkConfig {
            name: "Medium Dataset (50x5)".to_string(),
            n: 50,
            d: 5,
            scale: 100,
            epsilon: 1000.0,
        },
        BenchmarkConfig {
            name: "Large Dataset (100x7)".to_string(),
            n: 100,
            d: 7,
            scale: 100,
            epsilon: 1000.0,
        },
        BenchmarkConfig {
            name: "Iris-sized Dataset (150x4)".to_string(),
            n: 150,
            d: 4,
            scale: 100,
            epsilon: 10000.0,
        },
        BenchmarkConfig {
            name: "High Precision (50x5, scale=1000)".to_string(),
            n: 50,
            d: 5,
            scale: 1000,
            epsilon: 10000.0,
        },
    ]
}