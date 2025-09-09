use std::{
    io::{self, Write},
    time::Duration,
};
// Added for stdout flushing

use tokio::time::Instant;
use zuup_core::{DownloadRequestProps, ZuupConfig, ZuupEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable logging to see what's happening
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let config = ZuupConfig::default();
    let engine = ZuupEngine::new(config).await?;

    let magnet = "magnet:?xt=urn:btih:7920acc28f0ec4d75ac0e0edd82789092cf8ded1&dn=archlinux-2025.09.01-x86_64.iso";

    println!("Starting download: {}", magnet);

    let start = Instant::now();
    let request = DownloadRequestProps::new(vec![url::Url::parse(magnet)?]);
    let download_id = engine.add_download(request).await?;
    engine.start_download(&download_id).await?;

    loop {
        tokio::time::sleep(Duration::from_millis(250)).await;
        match engine.get_download_info(download_id.clone()).await {
            Ok(info) => {
                // All information is now provided by the library - no need to calculate!
                let downloaded = info.progress.downloaded_size;
                let total_size = info.progress.total_size;
                let speed = info.progress.download_speed;
                let upload_speed = info.progress.upload_speed.unwrap_or(0);
                let percentage = info.progress.percentage;
                let eta = info.progress.eta;
                let elapsed = start.elapsed();

                // Display progress information using library-provided data
                display_progress_enhanced(downloaded, total_size, speed, upload_speed, percentage, eta, elapsed);

                // Additional information from DownloadInfo
                if info.state.is_terminal() {
                    println!("\nDownload completed!");
                    println!(
                        "File saved to: {}/{}",
                        info.output_path.display(),
                        info.filename
                    );
                    if let Some(error) = &info.error_message {
                        println!("Error: {}", error);
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
    Ok(())
}

/// Display progress information using library-provided data
fn display_progress_enhanced(
    downloaded: u64, 
    total_size: Option<u64>, 
    speed: u64, 
    upload_speed: u64,
    percentage: u8,
    eta: Option<Duration>,
    elapsed: Duration
) {
    // Create progress bar using library-calculated percentage
    let (progress_bar, _) = if let Some(total) = total_size {
        let bar_width = 50;
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
        let bar = format!("[{}] ???%", "█".repeat(25));
        (bar, 0)
    };

    // Format sizes and speeds using library data
    let downloaded_str = format_bytes(downloaded);
    let total_str = total_size
        .map(format_bytes)
        .unwrap_or_else(|| "Unknown".to_string());
    let speed_str = format_bytes(speed);
    let upload_str = format_bytes(upload_speed);

    // Use library-calculated ETA
    let eta_str = eta
        .map(|d| format_duration(d.as_secs()))
        .unwrap_or_else(|| {
            if percentage >= 100 {
                "Complete".to_string()
            } else {
                "Unknown".to_string()
            }
        });

    // Calculate speed in Mbps for reference
    let download_mbps = (speed as f64 * 8.0) / 1_000_000.0;
    let upload_mbps = (upload_speed as f64 * 8.0) / 1_000_000.0;

    // Print progress with upload speed for torrents
    if upload_speed > 0 {
        print!(
            "\r{} {} / {} | ↓{}/s ({:.1} Mbps) ↑{}/s ({:.1} Mbps) | ETA: {} | Time: {}",
            progress_bar,
            downloaded_str,
            total_str,
            speed_str,
            download_mbps,
            upload_str,
            upload_mbps,
            eta_str,
            format_duration(elapsed.as_secs())
        );
    } else {
        print!(
            "\r{} {} / {} | {}/s ({:.1} Mbps) | ETA: {} | Time: {}",
            progress_bar,
            downloaded_str,
            total_str,
            speed_str,
            download_mbps,
            eta_str,
            format_duration(elapsed.as_secs())
        );
    }
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

/// Format bytes in human-readable format with better precision
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
