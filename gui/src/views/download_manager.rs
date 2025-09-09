//! Download manager view - primary interface component for managing downloads

use std::sync::Arc;
use gpui::*;
use gpui::prelude::FluentBuilder;
use gpui_component::{
    button::Button,
    Disableable,
    StyledExt,
};

use crate::{
    app::{ZuupApp, FilterState, SortState},
    views::{DownloadItemView, AddDownloadModal},
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
    
    /// Add download modal instance (when open)
    add_download_modal: Option<Entity<AddDownloadModal>>,
    
    /// Current filter state
    current_filter: FilterState,
    
    /// Current sort state
    current_sort: SortState,
}

impl DownloadManagerView {
    /// Create a new download manager view
    pub fn new(app: Arc<ZuupApp>) -> Self {
        let (current_filter, current_sort) = {
            let app_state = app.app_state().read().unwrap();
            (
                app_state.filter_state.clone(),
                app_state.sort_state.clone(),
            )
        };
        
        Self {
            app,
            filtered_downloads: Vec::new(),
            selected_downloads: std::collections::HashSet::new(),
            add_download_modal: None,
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
    fn handle_bulk_pause(&mut self, cx: &mut Context<Self>) {
        let selected_ids: Vec<_> = self.selected_downloads.iter().cloned().collect();
        let app = self.app.clone();
        
        tracing::info!("Bulk pause requested for {} downloads", selected_ids.len());
        
        tokio::spawn(async move {
            if let Some(adapter) = app.engine_adapter() {
                let mut success_count = 0;
                let mut error_count = 0;
                
                for download_id in selected_ids {
                    match adapter.engine().pause_download(&download_id).await {
                        Ok(()) => {
                            success_count += 1;
                            tracing::debug!("Successfully paused download: {}", download_id);
                        }
                        Err(e) => {
                            error_count += 1;
                            tracing::error!("Failed to pause download {}: {}", download_id, e);
                        }
                    }
                }
                
                tracing::info!("Bulk pause completed: {} successful, {} failed", success_count, error_count);
                
                if error_count > 0 {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some(format!("Failed to pause {} downloads", error_count));
                }
            } else {
                tracing::warn!("Engine adapter not available for bulk pause operation");
                {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some("Engine not available".to_string());
                }
            }
        });
    }
    
    /// Handle bulk resume action
    fn handle_bulk_resume(&mut self, cx: &mut Context<Self>) {
        let selected_ids: Vec<_> = self.selected_downloads.iter().cloned().collect();
        let app = self.app.clone();
        
        tracing::info!("Bulk resume requested for {} downloads", selected_ids.len());
        
        tokio::spawn(async move {
            if let Some(adapter) = app.engine_adapter() {
                let mut success_count = 0;
                let mut error_count = 0;
                
                for download_id in selected_ids {
                    match adapter.engine().resume_download(&download_id).await {
                        Ok(()) => {
                            success_count += 1;
                            tracing::debug!("Successfully resumed download: {}", download_id);
                        }
                        Err(e) => {
                            error_count += 1;
                            tracing::error!("Failed to resume download {}: {}", download_id, e);
                        }
                    }
                }
                
                tracing::info!("Bulk resume completed: {} successful, {} failed", success_count, error_count);
                
                if error_count > 0 {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some(format!("Failed to resume {} downloads", error_count));
                }
            } else {
                tracing::warn!("Engine adapter not available for bulk resume operation");
                {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some("Engine not available".to_string());
                }
            }
        });
    }
    
    /// Handle bulk cancel action
    fn handle_bulk_cancel(&mut self, cx: &mut Context<Self>) {
        let selected_ids: Vec<_> = self.selected_downloads.iter().cloned().collect();
        let app = self.app.clone();
        
        tracing::info!("Bulk cancel requested for {} downloads", selected_ids.len());
        
        // Show confirmation for bulk cancel
        // For now, proceed directly (in a full implementation, show confirmation dialog)
        tokio::spawn(async move {
            if let Some(adapter) = app.engine_adapter() {
                let mut success_count = 0;
                let mut error_count = 0;
                
                for download_id in selected_ids {
                    match adapter.engine().remove_download(&download_id, true).await {
                        Ok(()) => {
                            success_count += 1;
                            tracing::debug!("Successfully cancelled download: {}", download_id);
                        }
                        Err(e) => {
                            error_count += 1;
                            tracing::error!("Failed to cancel download {}: {}", download_id, e);
                        }
                    }
                }
                
                tracing::info!("Bulk cancel completed: {} successful, {} failed", success_count, error_count);
                
                if error_count > 0 {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some(format!("Failed to cancel {} downloads", error_count));
                }
            } else {
                tracing::warn!("Engine adapter not available for bulk cancel operation");
                {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some("Engine not available".to_string());
                }
            }
        });
    }
    
    /// Handle clear completed action
    fn handle_clear_completed(&mut self, cx: &mut Context<Self>) {
        let app = self.app.clone();
        
        tracing::info!("Clear completed downloads requested");
        
        tokio::spawn(async move {
            if let Some(adapter) = app.engine_adapter() {
                // Get all completed downloads
                match adapter.engine().completed_downloads().await {
                    Ok(completed_downloads) => {
                        let mut success_count = 0;
                        let mut error_count = 0;
                        
                        for download in completed_downloads {
                            match adapter.engine().remove_download(&download.id, false).await {
                                Ok(()) => {
                                    success_count += 1;
                                    tracing::debug!("Successfully removed completed download: {}", download.id);
                                }
                                Err(e) => {
                                    error_count += 1;
                                    tracing::error!("Failed to remove completed download {}: {}", download.id, e);
                                }
                            }
                        }
                        
                        tracing::info!("Clear completed downloads: {} successful, {} failed", success_count, error_count);
                        
                        if error_count > 0 {
                            let mut app_state = app.app_state().write().unwrap();
                            app_state.last_error = Some(format!("Failed to clear {} completed downloads", error_count));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to get completed downloads: {}", e);
                        {
                            let mut app_state = app.app_state().write().unwrap();
                            app_state.last_error = Some(format!("Failed to get completed downloads: {}", e));
                        }
                    }
                }
            } else {
                tracing::warn!("Engine adapter not available for clear completed operation");
                {
                    let mut app_state = app.app_state().write().unwrap();
                    app_state.last_error = Some("Engine not available".to_string());
                }
            }
        });
    }
    
    /// Handle opening the add download modal
    fn handle_open_add_download_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Update app state
        {
            let mut app_state = self.app.app_state().write().unwrap();
            app_state.add_download_modal_open = true;
        }
        
        // Create the modal
        self.add_download_modal = Some(cx.new(|cx| {
            AddDownloadModal::new(self.app.clone(), window, cx)
        }));
        
        tracing::info!("Add download modal opened");
        cx.notify();
    }
    
    /// Handle closing the add download modal
    fn handle_close_add_download_modal(&mut self, cx: &mut Context<Self>) {
        // Update app state
        {
            let mut app_state = self.app.app_state().write().unwrap();
            app_state.add_download_modal_open = false;
        }
        
        // Remove the modal
        self.add_download_modal = None;
        
        tracing::info!("Add download modal closed");
        cx.notify();
    }
    
    /// Check if the add download modal should be open
    fn should_show_add_download_modal(&self) -> bool {
        let app_state = self.app.app_state().read().unwrap();
        app_state.add_download_modal_open
    }
    
    /// Render the toolbar with action buttons
    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                    .on_click(cx.listener(|this, _event, window, cx| {
                        this.handle_open_add_download_modal(window, cx);
                    }))
            )
            .child(
                Button::new("pause_all")
                    .label("Pause All")
                    .disabled(self.selected_downloads.is_empty())
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        this.handle_bulk_pause(cx);
                    }))
            )
            .child(
                Button::new("resume_all")
                    .label("Resume All")
                    .disabled(self.selected_downloads.is_empty())
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        this.handle_bulk_resume(cx);
                    }))
            )
            .child(
                Button::new("clear_completed")
                    .label("Clear Completed")
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        this.handle_clear_completed(cx);
                    }))
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
    fn render_filter_tabs(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.handle_filter_change(FilterState::All, cx);
                            }))
                    )
                    .child(
                        Button::new("filter_active")
                            .label("Active")
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.handle_filter_change(FilterState::Active, cx);
                            }))
                    )
                    .child(
                        Button::new("filter_completed")
                            .label("Completed")
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.handle_filter_change(FilterState::Completed, cx);
                            }))
                    )
                    .child(
                        Button::new("filter_failed")
                            .label("Failed")
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.handle_filter_change(FilterState::Failed, cx);
                            }))
                    )
                    .child(
                        Button::new("filter_paused")
                            .label("Paused")
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.handle_filter_change(FilterState::Paused, cx);
                            }))
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
        // Check if modal state has changed and update accordingly
        let should_show_modal = self.should_show_add_download_modal();
        let has_modal = self.add_download_modal.is_some();
        
        if should_show_modal && !has_modal {
            self.handle_open_add_download_modal(_window, cx);
        } else if !should_show_modal && has_modal {
            self.handle_close_add_download_modal(cx);
        }
        
        div()
            .v_flex()
            .size_full()
            .child(self.render_toolbar(cx))
            .child(self.render_filter_tabs(cx))
            .child(self.render_download_list(cx))
            .child(self.render_status_bar(cx))
            .when_some(self.add_download_modal.clone(), |this, modal| {
                this.child(modal)
            })
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