//! HTTP/HTTPS Multi-Source Download Example
//!
//! This example demonstrates how to use Zuup Core with multiple HTTP/HTTPS sources
//! for the same file. It showcases the multi-source download capability using
//! multiple Arch Linux ISO mirrors for redundancy and potentially faster downloads.
//!
//! todo)) add actual multi implementation
//!
//! To run this example:
//! ```bash
//! cargo run --example http_multisource_download --features http
//! ```

use std::io::{self, Write};
use std::time::Duration;
use tokio::time::Instant;
use zuup_core::{ZuupConfig, ZuupEngine, types::DownloadRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with more detailed output for multi-source downloads
    tracing_subscriber::fmt()
        .with_env_filter("zuup_core=info,zuup_core::protocols::http=debug,zuup_core::engine=debug")
        .init();

    let config = ZuupConfig::default();
    let engine = ZuupEngine::new(config).await?;

    // Multiple Arch Linux ISO mirrors for the same file
    let arch_iso_mirrors = vec![
        "https://geo.mirror.pkgbuild.com/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://al.arch.niranjan.co/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://mirror.aarnet.edu.au/pub/archlinux/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://mirror.alwyzon.net/archlinux/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://mirror.ourhost.az/archlinux/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://mirror.1ago.be/archlinux/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://archlinux.c3sl.ufpr.br/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
        "https://mirror.telepoint.bg/archlinux/iso/2025.09.01/archlinux-2025.09.01-x86_64.iso",
    ];

    println!("Starting multi-source download with {} mirrors:", arch_iso_mirrors.len());
    for (i, mirror) in arch_iso_mirrors.iter().enumerate() {
        println!("  {}. {}", i + 1, mirror);
    }
    println!();

    let start = Instant::now();
    
    // Parse URLs and create download request with multiple sources
    let urls: Result<Vec<_>, _> = arch_iso_mirrors
        .iter()
        .map(|url| url::Url::parse(url))
        .collect();
    
    let urls = urls?;
    let request = DownloadRequest::new(urls);
    let download_id = engine.add_download(request).await?;
    
    println!("Download ID: {}", download_id);
    engine.start_download(&download_id).await?;

    let mut last_active_sources = 0;
    
    // Monitor download progress
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        match engine.get_download_info(download_id.clone()).await {
            Ok(info) => {
                let downloaded = info.progress.downloaded_size;
                let total_size = info.progress.total_size;
                let active_sources = info.progress.connections;

                // Calculate elapsed time and speed
                let elapsed = start.elapsed();
                let speed = if elapsed.as_secs() > 0 {
                    downloaded / elapsed.as_secs()
                } else {
                    0
                };

                // Show source changes
                if active_sources != last_active_sources {
                    println!("\nActive sources changed: {} -> {}", last_active_sources, active_sources);
                    last_active_sources = active_sources;
                }

                // Display progress with multi-source information
                display_multisource_progress(
                    downloaded, 
                    total_size, 
                    speed, 
                    elapsed, 
                    active_sources,
                    arch_iso_mirrors.len()
                );

                if info.state.is_terminal() {
                    println!("\n\nMulti-source download completed!");
                    println!("File saved to: {}/{}", info.output_path.display(), info.filename);
                    
                    if let Some(error) = &info.error_message {
                        println!("Error: {}", error);
                    } else {
                        println!("Successfully downloaded using {} mirror sources", arch_iso_mirrors.len());
                        println!("Total time: {}", format_duration(elapsed.as_secs()));
                        println!("Average speed: {}/s", format_bytes(speed));
                    }
                    break;
                }
            }
            Err(e) => {
                eprintln!("\nError getting download info: {}", e);
                break;
            }
        }
    }

    engine.shutdown(false).await?;
    Ok(())
}

/// Display progress information with multi-source details
fn display_multisource_progress(
    downloaded: u64, 
    total_size: Option<u64>, 
    speed: u64, 
    elapsed: Duration,
    active_sources: u32,
    total_sources: usize,
) {
    // Create progress bar
    let (progress_bar, _percentage) = if let Some(total) = total_size {
        let percentage = if total > 0 {
            (downloaded as f64 / total as f64 * 100.0) as u8
        } else {
            0
        };
        let bar_width = 40;
        let filled = (percentage as usize * bar_width) / 100;
        let empty = bar_width - filled;
        let bar = format!(
            "[{}{}] {:3}%",
            "█".repeat(filled),
            "░".repeat(empty),
            percentage
        );
        (bar, percentage)
    } else {
        let bar = format!("[{}] ???%", "█".repeat(20));
        (bar, 0)
    };

    // Format sizes and speeds
    let downloaded_str = format_bytes(downloaded);
    let total_str = total_size
        .map(format_bytes)
        .unwrap_or_else(|| "Unknown".to_string());
    let speed_str = format_bytes(speed);

    // Calculate ETA
    let eta = if speed > 0 && total_size.is_some() {
        let remaining = total_size.unwrap().saturating_sub(downloaded);
        if remaining > 0 {
            let eta_secs = remaining / speed;
            format_duration(eta_secs)
        } else {
            "Complete".to_string()
        }
    } else {
        "Unknown".to_string()
    };

    // Calculate speed in Mbps
    let mbps = (speed as f64 * 8.0) / 1_000_000.0;

    // Source indicator
    let source_indicator = if active_sources > 0 {
        format!("📡 {}/{}", active_sources, total_sources)
    } else {
        format!("⏸️  0/{}", total_sources)
    };

    // Print progress with multi-source info
    print!(
        "\r{} {} / {} | {}/s ({:.1} Mbps) | {} | ETA: {} | {}",
        progress_bar,
        downloaded_str,
        total_str,
        speed_str,
        mbps,
        source_indicator,
        eta,
        format_duration(elapsed.as_secs())
    );
    io::stdout().flush().unwrap();
}

/// Format duration in human-readable format
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Format bytes in human-readable format
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut size = bytes as f64;
    let mut unit_index = 0;
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else if size >= 100.0 {
        format!("{:.0} {}", size, UNITS[unit_index])
    } else if size >= 10.0 {
        format!("{:.1} {}", size, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}