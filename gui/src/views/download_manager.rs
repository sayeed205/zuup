//! Download manager view - primary interface component for managing downloads

use std::sync::Arc;
use gpui::*;
use gpui_component::{
    button::Button,
    Disableable,
    StyledExt,
};

use crate::{
    app::{ZuupApp, FilterState, SortState},
    views::DownloadItemView,
};
use zuup_core::types::{DownloadInfo, DownloadState};

/// Primary download management interface component
pub struct DownloadManagerView {
    /// Reference to the main application
    app: Arc<ZuupApp>,
    
    /// List of downloads to display (filtered and sorted)
    filtered_downloads: Vec<DownloadInfo>,
    
    /// Currently selected download IDs
    selected_downloads: std::collections::HashSet<zuup_core::types::DownloadId>,
    
    /// Whether the add download modal is open
    add_download_modal_open: bool,
    
    /// Current filter state
    current_filter: FilterState,
    
    /// Current sort state
    current_sort: SortState,
}

impl DownloadManagerView {
    /// Create a new download manager view
    pub fn new(app: Arc<ZuupApp>) -> Self {
        let (add_download_modal_open, current_filter, current_sort) = {
            let app_state = app.app_state().read().unwrap();
            (
                app_state.add_download_modal_open,
                app_state.filter_state.clone(),
                app_state.sort_state.clone(),
            )
        };
        
        Self {
            app,
            filtered_downloads: Vec::new(),
            selected_downloads: std::collections::HashSet::new(),
            add_download_modal_open,
            current_filter,
            current_sort,
        }
    }
    
    /// Update the download list from the core engine
    pub fn update_downloads(&mut self, downloads: Vec<DownloadInfo>) {
        // Apply current filter
        let filtered: Vec<DownloadInfo> = downloads
            .into_iter()
            .filter(|download| self.matches_filter(download))
            .collect();
        
        // Apply current sort
        let mut sorted = filtered;
        self.sort_downloads(&mut sorted);
        
        self.filtered_downloads = sorted;
    }
    
    /// Check if a download matches the current filter
    fn matches_filter(&self, download: &DownloadInfo) -> bool {
        match self.current_filter {
            FilterState::All => true,
            FilterState::Active => matches!(
                download.state,
                DownloadState::Active | DownloadState::Preparing
            ),
            FilterState::Completed => matches!(download.state, DownloadState::Completed),
            FilterState::Failed => matches!(download.state, DownloadState::Failed(_)),
            FilterState::Paused => matches!(download.state, DownloadState::Paused),
        }
    }
    
    /// Sort downloads according to current sort state
    fn sort_downloads(&self, downloads: &mut Vec<DownloadInfo>) {
        match self.current_sort {
            SortState::Name => {
                downloads.sort_by(|a, b| a.filename.cmp(&b.filename));
            }
            SortState::Size => {
                downloads.sort_by(|a, b| {
                    let a_size = a.file_size.unwrap_or(0);
                    let b_size = b.file_size.unwrap_or(0);
                    b_size.cmp(&a_size) // Descending order
                });
            }
            SortState::Progress => {
                downloads.sort_by(|a, b| {
                    b.progress.percentage.cmp(&a.progress.percentage) // Descending order
                });
            }
            SortState::Speed => {
                downloads.sort_by(|a, b| {
                    b.progress.download_speed.cmp(&a.progress.download_speed) // Descending order
                });
            }
            SortState::DateAdded => {
                downloads.sort_by(|a, b| b.created_at.cmp(&a.created_at)); // Most recent first
            }
        }
    }
    
    /// Handle filter change
    fn handle_filter_change(&mut self, new_filter: FilterState, _cx: &mut Context<Self>) {
        self.current_filter = new_filter.clone();
        
        // Update app state
        {
            let mut app_state = self.app.app_state().write().unwrap();
            app_state.filter_state = new_filter;
        }
        
        // Refresh the download list with new filter
        // In a real implementation, this would trigger a refresh from the engine
        tracing::debug!("Filter changed to {:?}", self.current_filter);
    }
    
    /// Handle sort change
    fn handle_sort_change(&mut self, new_sort: SortState, _cx: &mut Context<Self>) {
        self.current_sort = new_sort.clone();
        
        // Update app state
        {
            let mut app_state = self.app.app_state().write().unwrap();
            app_state.sort_state = new_sort;
        }
        
        // Re-sort current downloads
        let mut downloads = self.filtered_downloads.clone();
        self.sort_downloads(&mut downloads);
        self.filtered_downloads = downloads;
        
        tracing::debug!("Sort changed to {:?}", self.current_sort);
    }
    
    /// Handle download selection
    fn handle_download_selection(&mut self, download_id: zuup_core::types::DownloadId, selected: bool, _cx: &mut Context<Self>) {
        if selected {
            self.selected_downloads.insert(download_id);
        } else {
            self.selected_downloads.remove(&download_id);
        }
        
        tracing::debug!("Download selection changed: {} downloads selected", self.selected_downloads.len());
    }
    
    /// Handle bulk pause action
    fn handle_bulk_pause(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Bulk pause requested for {} downloads", self.selected_downloads.len());
        // TODO: Implement bulk pause via engine adapter
    }
    
    /// Handle bulk resume action
    fn handle_bulk_resume(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Bulk resume requested for {} downloads", self.selected_downloads.len());
        // TODO: Implement bulk resume via engine adapter
    }
    
    /// Handle bulk cancel action
    fn handle_bulk_cancel(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Bulk cancel requested for {} downloads", self.selected_downloads.len());
        // TODO: Implement bulk cancel via engine adapter
    }
    
    /// Handle clear completed action
    fn handle_clear_completed(&mut self, _cx: &mut Context<Self>) {
        tracing::info!("Clear completed downloads requested");
        // TODO: Implement clear completed via engine adapter
    }
    
    /// Render the toolbar with action buttons
    fn render_toolbar(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .h_flex()
            .gap_2()
            .p_3()
            .bg(gpui::rgb(0xffffff))
            .border_b_1()
            .border_color(gpui::rgb(0xd1d5db))
            .child(
                Button::new("add_download")
                    .label("Add Download")
                    .on_click(move |_, _, _cx| {
                        tracing::info!("Add download button clicked");
                        // TODO: Open add download modal
                    })
            )
            .child(
                Button::new("pause_all")
                    .label("Pause All")
                    .disabled(self.selected_downloads.is_empty())
                    .on_click(move |_, _, _cx| {
                        tracing::info!("Pause all button clicked");
                        // TODO: Implement pause all
                    })
            )
            .child(
                Button::new("resume_all")
                    .label("Resume All")
                    .disabled(self.selected_downloads.is_empty())
                    .on_click(move |_, _, _cx| {
                        tracing::info!("Resume all button clicked");
                        // TODO: Implement resume all
                    })
            )
            .child(
                Button::new("clear_completed")
                    .label("Clear Completed")
                    .on_click(move |_, _, _cx| {
                        tracing::info!("Clear completed button clicked");
                        // TODO: Implement clear completed
                    })
            )
            .child(
                Button::new("settings")
                    .label("Settings")
                    .on_click(move |_, _, _cx| {
                        tracing::info!("Settings button clicked");
                        // TODO: Open settings modal
                    })
            )
    }
    
    /// Render the filter tabs
    fn render_filter_tabs(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let _current_filter = &self.current_filter;
        
        div()
            .h_flex()
            .p_2()
            .bg(gpui::rgb(0xf9fafb))
            .border_b_1()
            .border_color(gpui::rgb(0xd1d5db))
            .child(
                div()
                    .h_flex()
                    .gap_1()
                    .child(
                        Button::new("filter_all")
                            .label("All")
                            .on_click(move |_, _, _cx| {
                                tracing::debug!("All filter tab clicked");
                                // TODO: Handle filter change to All
                            })
                    )
                    .child(
                        Button::new("filter_active")
                            .label("Active")
                            .on_click(move |_, _, _cx| {
                                tracing::debug!("Active filter tab clicked");
                                // TODO: Handle filter change to Active
                            })
                    )
                    .child(
                        Button::new("filter_completed")
                            .label("Completed")
                            .on_click(move |_, _, _cx| {
                                tracing::debug!("Completed filter tab clicked");
                                // TODO: Handle filter change to Completed
                            })
                    )
                    .child(
                        Button::new("filter_failed")
                            .label("Failed")
                            .on_click(move |_, _, _cx| {
                                tracing::debug!("Failed filter tab clicked");
                                // TODO: Handle filter change to Failed
                            })
                    )
                    .child(
                        Button::new("filter_paused")
                            .label("Paused")
                            .on_click(move |_, _, _cx| {
                                tracing::debug!("Paused filter tab clicked");
                                // TODO: Handle filter change to Paused
                            })
                    )
            )
    }
    
    /// Render the download list
    fn render_download_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        if self.filtered_downloads.is_empty() {
            return div()
                .flex_1()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::rgb(0xffffff))
                .text_color(gpui::rgb(0x6b7280))
                .text_lg()
                .child(match self.current_filter {
                    FilterState::All => "No downloads yet. Click 'Add Download' to get started.",
                    FilterState::Active => "No active downloads.",
                    FilterState::Completed => "No completed downloads.",
                    FilterState::Failed => "No failed downloads.",
                    FilterState::Paused => "No paused downloads.",
                })
                .into_any_element();
        }
        
        div()
            .flex_1()
            .overflow_hidden()
            .bg(gpui::rgb(0xffffff))
            .child(
                div()
                    .v_flex()
                    .children(
                        self.filtered_downloads
                            .iter()
                            .map(|download| {
                                let download_id = download.id.clone();
                                let is_selected = self.selected_downloads.contains(&download_id);
                                
                                cx.new(|cx| DownloadItemView::new(
                                    download.clone(),
                                    is_selected,
                                    self.app.clone(),
                                    cx,
                                ))
                            })
                            .collect::<Vec<_>>()
                    )
            )
            .into_any_element()
    }
    
    /// Render the status bar
    fn render_status_bar(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let active_count = self.filtered_downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Active | DownloadState::Preparing))
            .count();
        
        let completed_count = self.filtered_downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Completed))
            .count();
        
        let total_speed: u64 = self.filtered_downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .map(|d| d.progress.download_speed)
            .sum();
        
        let speed_text = if total_speed > 0 {
            format_bytes_per_second(total_speed)
        } else {
            "0 B/s".to_string()
        };
        
        div()
            .h_flex()
            .justify_between()
            .items_center()
            .px_3()
            .py_2()
            .bg(gpui::rgb(0xf9fafb))
            .border_t_1()
            .border_color(gpui::rgb(0xd1d5db))
            .text_sm()
            .text_color(gpui::rgb(0x6b7280))
            .child(
                div()
                    .h_flex()
                    .gap_4()
                    .child(format!("{} active", active_count))
                    .child(format!("{} completed", completed_count))
                    .child(format!("{} total", self.filtered_downloads.len()))
            )
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .child("Total speed:")
                    .child(speed_text)
            )
    }
}

impl Render for DownloadManagerView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .v_flex()
            .size_full()
            .child(self.render_toolbar(cx))
            .child(self.render_filter_tabs(cx))
            .child(self.render_download_list(cx))
            .child(self.render_status_bar(cx))
    }
}

/// Format bytes per second for display
fn format_bytes_per_second(bytes_per_second: u64) -> String {
    const UNITS: &[&str] = &["B/s", "KB/s", "MB/s", "GB/s", "TB/s"];
    
    if bytes_per_second == 0 {
        return "0 B/s".to_string();
    }
    
    let mut size = bytes_per_second as f64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use zuup_core::types::{DownloadId, DownloadProgress, DownloadState};
    use chrono::Utc;
    use std::path::PathBuf;
    use url::Url;
    
    fn create_test_download(filename: &str, state: DownloadState) -> DownloadInfo {
        DownloadInfo {
            id: DownloadId::new(),
            urls: vec![Url::parse("http://example.com/file").unwrap()],
            filename: filename.to_string(),
            output_path: PathBuf::from("/tmp"),
            state,
            progress: DownloadProgress::new(),
            priority: zuup_core::types::DownloadPriority::Normal,
            download_type: zuup_core::types::DownloadType::Standard,
            category: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            options: zuup_core::types::DownloadOptions::default(),
            error_message: None,
            file_size: Some(1024 * 1024), // 1MB
            content_type: None,
            last_modified: None,
            referrer: None,
            cookies: None,
        }
    }
    
    #[test]
    fn test_filter_matching() {
        let app = Arc::new(crate::app::ZuupApp::new_simple().unwrap());
        let mut view = DownloadManagerView::new(app);
        
        // Test active filter
        view.current_filter = FilterState::Active;
        let active_download = create_test_download("active.zip", DownloadState::Active);
        let completed_download = create_test_download("completed.zip", DownloadState::Completed);
        
        assert!(view.matches_filter(&active_download));
        assert!(!view.matches_filter(&completed_download));
        
        // Test completed filter
        view.current_filter = FilterState::Completed;
        assert!(!view.matches_filter(&active_download));
        assert!(view.matches_filter(&completed_download));
    }
    
    #[test]
    fn test_download_sorting() {
        let app = Arc::new(crate::app::ZuupApp::new_simple().unwrap());
        let mut view = DownloadManagerView::new(app);
        
        let mut downloads = vec![
            create_test_download("z_file.zip", DownloadState::Active),
            create_test_download("a_file.zip", DownloadState::Active),
            create_test_download("m_file.zip", DownloadState::Active),
        ];
        
        // Test name sorting
        view.current_sort = SortState::Name;
        view.sort_downloads(&mut downloads);
        
        assert_eq!(downloads[0].filename, "a_file.zip");
        assert_eq!(downloads[1].filename, "m_file.zip");
        assert_eq!(downloads[2].filename, "z_file.zip");
    }
    
    #[test]
    fn test_format_bytes_per_second() {
        assert_eq!(format_bytes_per_second(0), "0 B/s");
        assert_eq!(format_bytes_per_second(512), "512 B/s");
        assert_eq!(format_bytes_per_second(1024), "1.0 KB/s");
        assert_eq!(format_bytes_per_second(1536), "1.5 KB/s");
        assert_eq!(format_bytes_per_second(1024 * 1024), "1.0 MB/s");
        assert_eq!(format_bytes_per_second(1024 * 1024 * 1024), "1.0 GB/s");
    }
}