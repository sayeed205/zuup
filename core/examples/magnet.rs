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
                // Extract information from DownloadInfo
                let downloaded = info.progress.downloaded_size;
                let total_size = info.progress.total_size;

                // Calculate elapsed time and speed
                let elapsed = start.elapsed();
                let speed = if elapsed.as_secs() > 0 {
                    downloaded / elapsed.as_secs()
                } else {
                    0
                };

                // Display progress information
                display_progress(downloaded, total_size, speed, elapsed);

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

/// Display progress information with enhanced formatting
fn display_progress(downloaded: u64, total_size: Option<u64>, speed: u64, elapsed: Duration) {
    // Create progress bar
    let (progress_bar, percentage) = if let Some(total) = total_size {
        let percentage = if total > 0 {
            (downloaded as f64 / total as f64 * 100.0) as u8
        } else {
            0
        };
        let bar_width = 50; // Wider progress bar
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

    // Calculate speed in Mbps for reference
    let mbps = (speed as f64 * 8.0) / 1_000_000.0;

    // Print progress (overwrite previous line)
    print!(
        "\r{} {} / {} | {}/s ({:.1} Mbps) | ETA: {} | Time: {}",
        progress_bar,
        downloaded_str,
        total_str,
        speed_str,
        mbps,
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
