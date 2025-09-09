//! Download item view component with progress bar and action buttons

use std::sync::Arc;
use gpui::*;
use gpui_component::{
    button::Button,
    Disableable,
    StyledExt,
};

use crate::app::ZuupApp;
use zuup_core::types::{DownloadInfo, DownloadState};

/// Individual download item component with progress and controls
pub struct DownloadItemView {
    /// Download information
    download_info: DownloadInfo,
    
    /// Whether this item is selected
    is_selected: bool,
    
    /// Reference to the main application
    app: Arc<ZuupApp>,
    
    /// Whether the context menu is open
    context_menu_open: bool,
    
    /// Last update timestamp for progress animation
    last_update: std::time::Instant,
}

impl DownloadItemView {
    /// Create a new download item view
    pub fn new(
        download_info: DownloadInfo,
        is_selected: bool,
        app: Arc<ZuupApp>,
        _cx: &mut Context<Self>,
    ) -> Self {
        Self {
            download_info,
            is_selected,
            app,
            context_menu_open: false,
            last_update: std::time::Instant::now(),
        }
    }
    
    /// Update the download information
    pub fn update_download_info(&mut self, new_info: DownloadInfo) {
        self.download_info = new_info;
        self.last_update = std::time::Instant::now();
    }
    
    /// Set selection state
    pub fn set_selected(&mut self, selected: bool) {
        self.is_selected = selected;
    }
    
    /// Handle pause action
    fn handle_pause(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Pause requested for download: {}", self.download_info.filename);
        // TODO: Implement pause via engine adapter
        // self.app.engine_adapter().pause_download(&self.download_info.id).await;
    }
    
    /// Handle resume action
    fn handle_resume(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Resume requested for download: {}", self.download_info.filename);
        // TODO: Implement resume via engine adapter
        // self.app.engine_adapter().resume_download(&self.download_info.id).await;
    }
    
    /// Handle cancel action
    fn handle_cancel(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Cancel requested for download: {}", self.download_info.filename);
        // TODO: Implement cancel via engine adapter
        // self.app.engine_adapter().cancel_download(&self.download_info.id).await;
    }
    
    /// Handle remove action
    fn handle_remove(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Remove requested for download: {}", self.download_info.filename);
        // TODO: Show confirmation dialog and implement remove via engine adapter
        // self.app.engine_adapter().remove_download(&self.download_info.id).await;
    }
    
    /// Handle open file action
    fn handle_open_file(&mut self, _cx: &mut Context<Self>) {
        if matches!(self.download_info.state, DownloadState::Completed) {
            let file_path = self.download_info.file_path();
            tracing::info!("Open file requested: {}", file_path.display());
            
            // TODO: Open file with system default application
            #[cfg(target_os = "linux")]
            {
                let _ = std::process::Command::new("xdg-open")
                    .arg(&file_path)
                    .spawn();
            }
            
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open")
                    .arg(&file_path)
                    .spawn();
            }
            
            #[cfg(target_os = "windows")]
            {
                let _ = std::process::Command::new("cmd")
                    .args(&["/C", "start", "", &file_path.to_string_lossy()])
                    .spawn();
            }
        }
    }
    
    /// Handle open folder action
    fn handle_open_folder(&mut self, _cx: &mut Context<Self>) {
        let folder_path = &self.download_info.output_path;
        tracing::info!("Open folder requested: {}", folder_path.display());
        
        // TODO: Open folder in file manager
        #[cfg(target_os = "linux")]
        {
            let _ = std::process::Command::new("xdg-open")
                .arg(folder_path)
                .spawn();
        }
        
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("open")
                .arg(folder_path)
                .spawn();
        }
        
        #[cfg(target_os = "windows")]
        {
            let _ = std::process::Command::new("explorer")
                .arg(folder_path)
                .spawn();
        }
    }
    
    /// Handle copy URL action
    fn handle_copy_url(&mut self, _cx: &mut Context<Self>) {
        if let Some(url) = self.download_info.urls.first() {
            tracing::info!("Copy URL requested: {}", url);
            // TODO: Copy URL to clipboard
            // This would require clipboard integration
        }
    }
    
    /// Handle retry action (for failed downloads)
    fn handle_retry(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Retry requested for download: {}", self.download_info.filename);
        // TODO: Implement retry via engine adapter
        // self.app.engine_adapter().retry_download(&self.download_info.id).await;
    }
    
    /// Get the appropriate action buttons based on download state
    fn get_action_buttons(&self) -> Vec<(&'static str, &'static str, bool)> {
        match &self.download_info.state {
            DownloadState::Active | DownloadState::Preparing => {
                vec![
                    ("pause", "Pause", true),
                    ("cancel", "Cancel", true),
                    ("open_folder", "Open Folder", true),
                ]
            }
            DownloadState::Paused => {
                vec![
                    ("resume", "Resume", true),
                    ("cancel", "Cancel", true),
                    ("open_folder", "Open Folder", true),
                ]
            }
            DownloadState::Completed => {
                vec![
                    ("open_file", "Open", true),
                    ("open_folder", "Open Folder", true),
                    ("remove", "Remove", true),
                ]
            }
            DownloadState::Failed(_) => {
                vec![
                    ("retry", "Retry", true),
                    ("remove", "Remove", true),
                    ("open_folder", "Open Folder", true),
                ]
            }
            DownloadState::Cancelled => {
                vec![
                    ("remove", "Remove", true),
                    ("open_folder", "Open Folder", true),
                ]
            }
            DownloadState::Pending | DownloadState::Waiting => {
                vec![
                    ("cancel", "Cancel", true),
                ]
            }
            DownloadState::Retrying => {
                vec![
                    ("cancel", "Cancel", true),
                    ("open_folder", "Open Folder", true),
                ]
            }
        }
    }
    
    /// Render the progress bar
    fn render_progress_bar(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let progress_value = if let Some(total_size) = self.download_info.progress.total_size {
            if total_size > 0 {
                (self.download_info.progress.downloaded_size as f64 / total_size as f64) * 100.0
            } else {
                0.0
            }
        } else {
            // Indeterminate progress for unknown size
            0.0
        };
        
        let progress_color = match &self.download_info.state {
            DownloadState::Active | DownloadState::Preparing => gpui::rgb(0x3b82f6), // Blue
            DownloadState::Completed => gpui::rgb(0x10b981), // Green
            DownloadState::Failed(_) => gpui::rgb(0xef4444), // Red
            DownloadState::Paused => gpui::rgb(0xf59e0b), // Amber
            DownloadState::Cancelled => gpui::rgb(0x6b7280), // Gray
            _ => gpui::rgb(0x6b7280), // Gray for other states
        };
        
        div()
            .w_full()
            .h_2()
            .bg(gpui::rgb(0xe5e7eb))
            .rounded_full()
            .overflow_hidden()
            .child(
                div()
                    .h_full()
                    .bg(progress_color)
                    .w(relative(progress_value as f32 / 100.0))
                    .rounded_full()
            )
    }
    
    /// Render download status and speed information
    fn render_status_info(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let status_text = match &self.download_info.state {
            DownloadState::Active => {
                let speed = format_bytes_per_second(self.download_info.progress.download_speed);
                let eta = if let Some(eta) = self.download_info.progress.eta {
                    format_duration(eta)
                } else {
                    "Unknown".to_string()
                };
                format!("{} • ETA: {}", speed, eta)
            }
            DownloadState::Preparing => "Preparing...".to_string(),
            DownloadState::Paused => "Paused".to_string(),
            DownloadState::Completed => {
                if let Some(completed_at) = self.download_info.completed_at {
                    let elapsed = chrono::Utc::now().signed_duration_since(completed_at);
                    format!("Completed {} ago", format_duration_chrono(elapsed))
                } else {
                    "Completed".to_string()
                }
            }
            DownloadState::Failed(error) => format!("Failed: {}", error),
            DownloadState::Cancelled => "Cancelled".to_string(),
            DownloadState::Pending => "Pending".to_string(),
            DownloadState::Waiting => "Waiting".to_string(),
            DownloadState::Retrying => "Retrying...".to_string(),
        };
        
        let size_text = if let Some(total_size) = self.download_info.progress.total_size {
            format!(
                "{} / {}",
                format_bytes(self.download_info.progress.downloaded_size),
                format_bytes(total_size)
            )
        } else {
            format_bytes(self.download_info.progress.downloaded_size)
        };
        
        div()
            .h_flex()
            .justify_between()
            .items_center()
            .text_sm()
            .child(
                div()
                    .text_color(gpui::rgb(0x6b7280))
                    .child(status_text)
            )
            .child(
                div()
                    .text_color(gpui::rgb(0x9ca3af))
                    .font_weight(FontWeight::MEDIUM)
                    .child(size_text)
            )
    }
    
    /// Render action buttons
    fn render_action_buttons(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let buttons = self.get_action_buttons();
        
        div()
            .h_flex()
            .gap_2()
            .children(
                buttons
                    .into_iter()
                    .map(|(action, label, enabled)| {
                        let download_id = self.download_info.id.clone();
                        Button::new(action)
                            .label(label)
                            .disabled(!enabled)
                            .on_click(move |_, _, _cx| {
                                match action {
                                    "pause" => {
                                        tracing::debug!("Pause button clicked for {}", download_id);
                                        // TODO: Handle pause
                                    }
                                    "resume" => {
                                        tracing::debug!("Resume button clicked for {}", download_id);
                                        // TODO: Handle resume
                                    }
                                    "cancel" => {
                                        tracing::debug!("Cancel button clicked for {}", download_id);
                                        // TODO: Handle cancel
                                    }
                                    "remove" => {
                                        tracing::debug!("Remove button clicked for {}", download_id);
                                        // TODO: Handle remove
                                    }
                                    "open_file" => {
                                        tracing::debug!("Open file button clicked for {}", download_id);
                                        // TODO: Handle open file
                                    }
                                    "open_folder" => {
                                        tracing::debug!("Open folder button clicked for {}", download_id);
                                        // TODO: Handle open folder
                                    }
                                    "retry" => {
                                        tracing::debug!("Retry button clicked for {}", download_id);
                                        // TODO: Handle retry
                                    }
                                    _ => {}
                                }
                            })
                    })
                    .collect::<Vec<_>>()
            )
    }
    
    /// Render context menu (placeholder for future implementation)
    fn _render_context_menu(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        // TODO: Implement context menu when gpui-component API is available
        div().child("Context menu placeholder")
    }
}

impl Render for DownloadItemView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.is_selected;
        let _download_id = self.download_info.id.clone();
        
        div()
            .w_full()
            .p_4()
            .border_b_1()
            .border_color(gpui::rgb(0xe5e7eb))
            .bg(if is_selected {
                gpui::rgb(0xdbeafe) // Light blue for selection
            } else {
                gpui::rgb(0xffffff) // White background
            })
            .hover(|style| style.bg(gpui::rgb(0xf3f4f6))) // Light gray on hover
            .child(
                div()
                    .v_flex()
                    .gap_2()
                    .child(
                        // Header row with filename and percentage
                        div()
                            .h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(gpui::rgb(0x111827))
                                    .text_base()
                                    .child(self.download_info.filename.clone())
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(gpui::rgb(0x6b7280))
                                    .font_weight(FontWeight::MEDIUM)
                                    .child(format!("{}%", self.download_info.progress.percentage))
                            )
                    )
                    .child(
                        // URL row
                        div()
                            .text_xs()
                            .text_color(gpui::rgb(0x9ca3af))
                            .child(
                                self.download_info.urls
                                    .first()
                                    .map(|url| url.to_string())
                                    .unwrap_or_else(|| "Unknown URL".to_string())
                            )
                    )
                    .child(
                        // Progress bar
                        self.render_progress_bar(cx)
                    )
                    .child(
                        // Status and action buttons row
                        div()
                            .h_flex()
                            .justify_between()
                            .items_center()
                            .child(self.render_status_info(cx))
                            .child(self.render_action_buttons(cx))
                    )
            )
    }
}

/// Format bytes for display
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
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Format bytes per second for display
fn format_bytes_per_second(bytes_per_second: u64) -> String {
    format!("{}/s", format_bytes(bytes_per_second))
}

/// Format duration for display
fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    
    if total_seconds < 60 {
        format!("{}s", total_seconds)
    } else if total_seconds < 3600 {
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{}m {}s", minutes, seconds)
    } else {
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        format!("{}h {}m", hours, minutes)
    }
}

/// Format chrono duration for display
fn format_duration_chrono(duration: chrono::Duration) -> String {
    if let Ok(std_duration) = duration.to_std() {
        format_duration(std_duration)
    } else {
        "Unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zuup_core::types::{DownloadId, DownloadProgress, DownloadState, DownloadPriority, DownloadType, DownloadOptions};
    use chrono::Utc;
    use std::path::PathBuf;
    use url::Url;
    
    fn create_test_download_info(state: DownloadState) -> DownloadInfo {
        DownloadInfo {
            id: DownloadId::new(),
            urls: vec![Url::parse("http://example.com/test.zip").unwrap()],
            filename: "test.zip".to_string(),
            output_path: PathBuf::from("/tmp"),
            state,
            progress: DownloadProgress::new(),
            priority: DownloadPriority::Normal,
            download_type: DownloadType::Standard,
            category: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            options: DownloadOptions::default(),
            error_message: None,
            file_size: Some(1024 * 1024), // 1MB
            content_type: None,
            last_modified: None,
            referrer: None,
            cookies: None,
        }
    }
    
    #[test]
    fn test_action_buttons_for_different_states() {
        let app = Arc::new(crate::app::ZuupApp::new_simple().unwrap());
        
        // Test active download buttons
        let active_info = create_test_download_info(DownloadState::Active);
        let view = DownloadItemView {
            download_info: active_info,
            is_selected: false,
            app: app.clone(),
            context_menu_open: false,
            last_update: std::time::Instant::now(),
        };
        
        let buttons = view.get_action_buttons();
        assert!(buttons.iter().any(|(action, _, _)| *action == "pause"));
        assert!(buttons.iter().any(|(action, _, _)| *action == "cancel"));
        
        // Test completed download buttons
        let completed_info = create_test_download_info(DownloadState::Completed);
        let view = DownloadItemView {
            download_info: completed_info,
            is_selected: false,
            app: app.clone(),
            context_menu_open: false,
            last_update: std::time::Instant::now(),
        };
        
        let buttons = view.get_action_buttons();
        assert!(buttons.iter().any(|(action, _, _)| *action == "open_file"));
        assert!(buttons.iter().any(|(action, _, _)| *action == "remove"));
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }
    
    #[test]
    fn test_format_duration() {
        use std::time::Duration;
        
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m");
    }
}