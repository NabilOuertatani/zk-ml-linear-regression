use crate::bench::BenchmarkResult;
use anyhow::Result;
use plotters::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
struct BenchmarkData {
    results: Vec<BenchmarkResult>,
}

/// Load benchmark results from JSON file
pub fn load_benchmark_results(filename: &str) -> Result<Vec<BenchmarkResult>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let data: BenchmarkData = serde_json::from_str(&contents)?;
    Ok(data.results)
}

/// Plot proving time vs dataset size
pub fn plot_proving_time(results: &[BenchmarkResult], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate data size (n * d) for x-axis
    let data_sizes: Vec<(usize, f64)> = results
        .iter()
        .filter(|r| r.scale == 100) // Only standard scale for fair comparison
        .map(|r| (r.n * r.d, r.proving_time_ms / 1000.0)) // Convert to seconds
        .collect();

    let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
    let max_time = data_sizes.iter().map(|(_, t)| *t).fold(0.0, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("ZK Proving Time vs Dataset Size", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0usize..max_size, 0f64..(max_time * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Dataset Size (n × d)")
        .y_desc("Proving Time (seconds)")
        .draw()?;

    // Draw points
    chart.draw_series(
        data_sizes
            .iter()
            .map(|&(size, time)| Circle::new((size, time), 5, RED.filled())),
    )?;

    // Draw connecting line
    chart.draw_series(LineSeries::new(data_sizes.clone(), &RED))?;

    root.present()?;
    println!(" Proving time plot saved to {}", filename);
    Ok(())
}

/// Plot verification time vs dataset size
pub fn plot_verification_time(results: &[BenchmarkResult], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let data_sizes: Vec<(usize, f64)> = results
        .iter()
        .filter(|r| r.scale == 100)
        .map(|r| (r.n * r.d, r.verification_time_ms))
        .collect();

    let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
    let max_time = data_sizes.iter().map(|(_, t)| *t).fold(0.0, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("ZK Verification Time vs Dataset Size", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0usize..max_size, 0f64..(max_time * 1.2))?;

    chart
        .configure_mesh()
        .x_desc("Dataset Size (n × d)")
        .y_desc("Verification Time (ms)")
        .draw()?;

    chart.draw_series(
        data_sizes
            .iter()
            .map(|&(size, time)| Circle::new((size, time), 5, BLUE.filled())),
    )?;

    chart.draw_series(LineSeries::new(data_sizes.clone(), &BLUE))?;

    root.present()?;
    println!(" Verification time plot saved to {}", filename);
    Ok(())
}

/// Plot proof size vs dataset size
pub fn plot_proof_size(results: &[BenchmarkResult], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let data_sizes: Vec<(usize, usize)> = results
        .iter()
        .filter(|r| r.scale == 100)
        .map(|r| (r.n * r.d, r.proof_size_bytes))
        .collect();

    let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
    let max_proof = data_sizes.iter().map(|(_, p)| *p).max().unwrap_or(300);

    let mut chart = ChartBuilder::on(&root)
        .caption("ZK Proof Size vs Dataset Size", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0usize..max_size, 0usize..(max_proof + 50))?;

    chart
        .configure_mesh()
        .x_desc("Dataset Size (n × d)")
        .y_desc("Proof Size (bytes)")
        .draw()?;

    chart.draw_series(
        data_sizes
            .iter()
            .map(|&(size, proof)| Circle::new((size, proof), 5, GREEN.filled())),
    )?;

    chart.draw_series(LineSeries::new(data_sizes.clone(), &GREEN))?;

    root.present()?;
    println!(" Proof size plot saved to {}", filename);
    Ok(())
}

/// Plot combined comparison
pub fn plot_scalability_comparison(results: &[BenchmarkResult], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    // Top: Proving time
    {
        let data_sizes: Vec<(usize, f64)> = results
            .iter()
            .filter(|r| r.scale == 100)
            .map(|r| (r.n * r.d, r.proving_time_ms / 1000.0))
            .collect();

        let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
        let max_time = data_sizes.iter().map(|(_, t)| *t).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&upper)
            .caption("Proving Time Scalability", ("sans-serif", 35))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0usize..max_size, 0f64..(max_time * 1.1))?;

        chart
            .configure_mesh()
            .x_desc("Dataset Size (n × d)")
            .y_desc("Proving Time (seconds)")
            .draw()?;

        chart.draw_series(
            data_sizes
                .iter()
                .map(|&(size, time)| Circle::new((size, time), 4, RED.filled())),
        )?;

        chart.draw_series(LineSeries::new(data_sizes, &RED.mix(0.8)))?;
    }

    // Bottom: Split for verification time and proof size
    let (lower_left, lower_right) = lower.split_horizontally(600);

    // Bottom-left: Verification time
    {
        let data_sizes: Vec<(usize, f64)> = results
            .iter()
            .filter(|r| r.scale == 100)
            .map(|r| (r.n * r.d, r.verification_time_ms))
            .collect();

        let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
        let max_time = data_sizes.iter().map(|(_, t)| *t).fold(0.0, f64::max);

        let mut chart = ChartBuilder::on(&lower_left)
            .caption("Verification Time", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0usize..max_size, 0f64..(max_time * 1.2))?;

        chart
            .configure_mesh()
            .x_desc("Dataset Size (n × d)")
            .y_desc("Time (ms)")
            .draw()?;

        chart.draw_series(
            data_sizes
                .iter()
                .map(|&(size, time)| Circle::new((size, time), 4, BLUE.filled())),
        )?;

        chart.draw_series(LineSeries::new(data_sizes, &BLUE.mix(0.8)))?;
    }

    // Bottom-right: Proof size
    {
        let data_sizes: Vec<(usize, usize)> = results
            .iter()
            .filter(|r| r.scale == 100)
            .map(|r| (r.n * r.d, r.proof_size_bytes))
            .collect();

        let max_size = data_sizes.iter().map(|(s, _)| *s).max().unwrap_or(1000);
        let max_proof = data_sizes.iter().map(|(_, p)| *p).max().unwrap_or(300);

        let mut chart = ChartBuilder::on(&lower_right)
            .caption("Proof Size", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0usize..max_size, 0usize..(max_proof + 50))?;

        chart
            .configure_mesh()
            .x_desc("Dataset Size (n × d)")
            .y_desc("Size (bytes)")
            .draw()?;

        chart.draw_series(
            data_sizes
                .iter()
                .map(|&(size, proof)| Circle::new((size, proof), 4, GREEN.filled())),
        )?;

        chart.draw_series(LineSeries::new(data_sizes, &GREEN.mix(0.8)))?;
    }

    root.present()?;
    println!(" Scalability comparison plot saved to {}", filename);
    Ok(())
}

/// Generate all benchmark plots
pub fn generate_all_plots(json_file: &str) -> Result<()> {
    println!("\n Generating benchmark visualization plots...\n");

    let results = load_benchmark_results(json_file)?;

    plot_proving_time(&results, "benchmark_proving_time.png")?;
    plot_verification_time(&results, "benchmark_verification_time.png")?;
    plot_proof_size(&results, "benchmark_proof_size.png")?;
    plot_scalability_comparison(&results, "benchmark_scalability.png")?;

    println!("\n All benchmark plots generated successfully!");
    Ok(())
}