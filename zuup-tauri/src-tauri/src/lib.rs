use tauri::Manager;
use tracing_subscriber;

#[cfg(target_os = "macos")]
use window_vibrancy::{NSVisualEffectMaterial, apply_vibrancy};

#[cfg(target_os = "windows")]
use window_vibrancy::apply_mica;

mod commands;
mod services;

use commands::*;
use services::{ConfigService, DownloadService, EventService};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_os::init())
        .setup(|app| {
            // Initialize services
            let config_service = ConfigService::new()
                .map_err(|e| format!("Failed to initialize config service: {}", e))?;

            let download_service = DownloadService::new();
            let event_service = EventService::new(app.handle().clone());

            // Store services in app state
            app.manage(config_service.clone());
            app.manage(download_service.clone());
            app.manage(event_service.clone());

            // Set up event service in download service
            download_service.set_event_service(event_service);

            let _window = app.get_webview_window("main").unwrap();

            #[cfg(target_os = "macos")]
            apply_vibrancy(&_window, NSVisualEffectMaterial::HudWindow, None, None)
                .expect("Unsupported platform! 'apply_vibrancy' is only supported on macOS");

            #[cfg(target_os = "windows")]
            apply_mica(&_window, None)
                .expect("Unsupported platform! 'apply_mica' is only supported on Windows");

            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            // Config commands
            load_config,
            save_config,
            get_gui_config,
            update_gui_config,
            // Download commands
            add_download,
            pause_download,
            resume_download,
            cancel_download,
            remove_download,
            get_downloads,
            get_download_stats
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
