//! Add download modal component for creating new download tasks

use std::path::PathBuf;
use std::sync::Arc;
use gpui::*;
use gpui::prelude::FluentBuilder;
use gpui_component::{
    button::{Button, ButtonVariant, ButtonVariants},
    input::{InputState, TextInput, InputEvent},
    modal::Modal,
    StyledExt,
};
use url::Url;
use zuup_core::types::DownloadRequest;

use crate::app::ZuupApp;

/// Supported protocols for download validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportedProtocol {
    Http,
    Https,
    Ftp,
    Sftp,
    Torrent,
}

impl SupportedProtocol {
    /// Get all supported protocols
    pub fn all() -> Vec<Self> {
        vec![
            Self::Http,
            Self::Https,
            Self::Ftp,
            Self::Sftp,
            Self::Torrent,
        ]
    }
    
    /// Get the protocol scheme string
    pub fn scheme(&self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
            Self::Ftp => "ftp",
            Self::Sftp => "sftp",
            Self::Torrent => "magnet", // Torrent uses magnet: scheme
        }
    }
    
    /// Check if a URL uses this protocol
    pub fn matches_url(&self, url: &Url) -> bool {
        match self {
            Self::Torrent => {
                // Handle both magnet links and .torrent file URLs
                url.scheme() == "magnet" || 
                (url.scheme() == "http" || url.scheme() == "https") && 
                url.path().ends_with(".torrent")
            }
            _ => url.scheme() == self.scheme(),
        }
    }
}

/// Validation result for URL input
#[derive(Debug, Clone)]
pub enum UrlValidationResult {
    Valid(Url, SupportedProtocol),
    Invalid(String),
    Empty,
}

/// Add download modal state and logic
pub struct AddDownloadModal {
    /// Reference to the main application
    app: Arc<ZuupApp>,
    
    /// URL input state
    url_input: Entity<InputState>,
    
    /// Filename input state
    filename_input: Entity<InputState>,
    
    /// Current URL validation result
    url_validation: UrlValidationResult,
    
    /// Selected destination folder
    destination_folder: Option<PathBuf>,
    
    /// Custom filename (optional)
    custom_filename: Option<String>,
    
    /// Whether the modal is currently submitting
    is_submitting: bool,
    
    /// Error message to display
    error_message: Option<String>,
    
    /// Focus handle for the modal
    focus_handle: FocusHandle,
}

impl AddDownloadModal {
    /// Create a new add download modal
    pub fn new(app: Arc<ZuupApp>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let url_input = cx.new(|cx| {
            InputState::new(window, cx)
        });
        
        let filename_input = cx.new(|cx| {
            InputState::new(window, cx)
        });
        
        let focus_handle = cx.focus_handle();
        
        // Subscribe to input events
        let _url_subscription = cx.subscribe_in(&url_input, window, Self::on_url_input_event);
        let _filename_subscription = cx.subscribe_in(&filename_input, window, Self::on_filename_input_event);
        
        Self {
            app,
            url_input,
            filename_input,
            url_validation: UrlValidationResult::Empty,
            destination_folder: None,
            custom_filename: None,
            is_submitting: false,
            error_message: None,
            focus_handle,
        }
    }
    
    /// Validate the current URL input
    fn validate_url(&mut self, url_text: &str) {
        if url_text.trim().is_empty() {
            self.url_validation = UrlValidationResult::Empty;
            return;
        }
        
        // Try to parse the URL
        match Url::parse(url_text.trim()) {
            Ok(url) => {
                // Check if the protocol is supported
                let supported_protocols = SupportedProtocol::all();
                
                for protocol in supported_protocols {
                    if protocol.matches_url(&url) {
                        self.url_validation = UrlValidationResult::Valid(url, protocol);
                        return;
                    }
                }
                
                // If we get here, the URL is valid but uses an unsupported protocol
                self.url_validation = UrlValidationResult::Invalid(
                    format!("Unsupported protocol '{}'. Supported protocols: HTTP, HTTPS, FTP, SFTP, and magnet links", url.scheme())
                );
            }
            Err(e) => {
                self.url_validation = UrlValidationResult::Invalid(
                    format!("Invalid URL: {}", e)
                );
            }
        }
    }
    
    /// Handle URL input event
    fn on_url_input_event(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let InputEvent::Change(text) = event {
            self.validate_url(text);
            self.error_message = None; // Clear any previous error
            cx.notify();
        }
    }
    
    /// Handle filename input event
    fn on_filename_input_event(
        &mut self,
        _: &Entity<InputState>,
        event: &InputEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let InputEvent::Change(text) = event {
            self.custom_filename = if text.trim().is_empty() {
                None
            } else {
                Some(text.to_string())
            };
            cx.notify();
        }
    }
    
    /// Open folder picker for destination selection
    fn open_folder_picker(&mut self, cx: &mut Context<Self>) {
        // Spawn async task to open native folder picker
        cx.spawn(async move |view: WeakEntity<Self>, cx| {
            let selected_folder = Self::show_folder_picker().await;
            
            if let Some(view) = view.upgrade() {
                let _ = view.update(cx, |this, cx| {
                    if let Some(folder) = selected_folder {
                        this.destination_folder = Some(folder);
                        tracing::info!("Folder selected: {:?}", this.destination_folder);
                    } else {
                        tracing::info!("Folder picker cancelled");
                    }
                    cx.notify();
                });
            }
        }).detach();
    }
    
    /// Show native folder picker dialog
    async fn show_folder_picker() -> Option<PathBuf> {
        // Use rfd (Rust File Dialog) for cross-platform native file dialogs
        tokio::task::spawn_blocking(|| {
            rfd::FileDialog::new()
                .set_title("Select Download Folder")
                .pick_folder()
        }).await.ok().flatten()
    }
    
    /// Handle form submission
    fn handle_submit(&mut self, url_text: String, cx: &mut Context<Self>) {
        // Validate URL first
        self.validate_url(&url_text);
        
        let (url, _protocol) = match &self.url_validation {
            UrlValidationResult::Valid(url, protocol) => (url.clone(), protocol.clone()),
            UrlValidationResult::Invalid(error) => {
                self.error_message = Some(error.clone());
                cx.notify();
                return;
            }
            UrlValidationResult::Empty => {
                self.error_message = Some("Please enter a URL".to_string());
                cx.notify();
                return;
            }
        };
        
        // Get destination folder (use default if not selected)
        let destination = self.destination_folder.clone().unwrap_or_else(|| {
            let config = self.app.config_manager().lock().unwrap();
            config.core_config().general.download_dir.clone()
        });
        
        // Create download request
        let mut request = DownloadRequest::new(vec![url]).output_path(destination);
        
        // Set custom filename if provided
        if let Some(filename) = &self.custom_filename {
            if !filename.trim().is_empty() {
                request = request.filename(filename.trim().to_string());
            }
        }
        
        self.is_submitting = true;
        cx.notify();
        
        // Submit the download request to the core engine
        let app = self.app.clone();
        cx.spawn(async move |view: WeakEntity<Self>, cx| {
            let result = app.add_download(request).await;
            
            if let Some(view) = view.upgrade() {
                let _ = view.update(cx, |this, cx| {
                    this.is_submitting = false;
                    
                    match result {
                        Ok(download_id) => {
                            tracing::info!("Download added successfully: {:?}", download_id);
                            this.close_modal(cx);
                        }
                        Err(error) => {
                            tracing::error!("Failed to add download: {}", error);
                            this.error_message = Some(format!("Failed to add download: {}", error));
                            cx.notify();
                        }
                    }
                });
            }
        }).detach();
    }
    
    /// Handle modal cancellation
    fn handle_cancel(&mut self, cx: &mut Context<Self>) {
        self.close_modal(cx);
    }
    
    /// Close the modal
    fn close_modal(&mut self, cx: &mut Context<Self>) {
        // Update app state to close the modal
        {
            let mut app_state = self.app.app_state().write().unwrap();
            app_state.add_download_modal_open = false;
        }
        
        tracing::info!("Add download modal closed");
        cx.notify();
    }
    
    /// Get the validation status message
    fn get_validation_message(&self) -> Option<String> {
        match &self.url_validation {
            UrlValidationResult::Valid(url, protocol) => {
                Some(format!("✓ Valid {} URL: {}", protocol.scheme().to_uppercase(), url))
            }
            UrlValidationResult::Invalid(error) => Some(format!("✗ {}", error)),
            UrlValidationResult::Empty => None,
        }
    }
    
    /// Check if the form is valid for submission
    fn is_form_valid(&self) -> bool {
        matches!(self.url_validation, UrlValidationResult::Valid(_, _)) && !self.is_submitting
    }
    
    /// Render the URL input section
    fn render_url_input(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .v_flex()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .font_medium()
                    .text_color(gpui::rgb(0x374151))
                    .child("Download URL")
            )
            .child(
                TextInput::new(&self.url_input)
            )
            .when_some(self.get_validation_message(), |this, message| {
                this.child(
                    div()
                        .text_xs()
                        .text_color(match &self.url_validation {
                            UrlValidationResult::Valid(_, _) => gpui::rgb(0x059669), // Green
                            UrlValidationResult::Invalid(_) => gpui::rgb(0xdc2626), // Red
                            UrlValidationResult::Empty => gpui::rgb(0x6b7280), // Gray
                        })
                        .child(message)
                )
            })
    }
    
    /// Render the destination folder section
    fn render_destination_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let folder_display = self.destination_folder
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| {
                let config = self.app.config_manager().lock().unwrap();
                format!("{} (default)", config.core_config().general.download_dir.display())
            });
        
        div()
            .v_flex()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .font_medium()
                    .text_color(gpui::rgb(0x374151))
                    .child("Destination Folder")
            )
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .flex_1()
                            .px_3()
                            .py_2()
                            .bg(gpui::rgb(0xf9fafb))
                            .border_1()
                            .border_color(gpui::rgb(0xd1d5db))
                            .rounded_md()
                            .text_sm()
                            .text_color(gpui::rgb(0x6b7280))
                            .child(folder_display)
                    )
                    .child(
                        Button::new("browse_folder")
                            .label("Browse")
                            .with_variant(ButtonVariant::Secondary)
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.open_folder_picker(cx);
                            }))
                    )
            )
    }
    
    /// Render the advanced options section
    fn render_advanced_options(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .v_flex()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .font_medium()
                    .text_color(gpui::rgb(0x374151))
                    .child("Advanced Options")
            )
            .child(
                div()
                    .v_flex()
                    .gap_2()
                    .child(
                        div()
                            .v_flex()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(gpui::rgb(0x6b7280))
                                    .child("Custom Filename (optional)")
                            )
                            .child(
                                TextInput::new(&self.filename_input)
                            )
                    )
            )
    }
    
    /// Render error message if any
    fn render_error_message(&self) -> Option<impl IntoElement> {
        self.error_message.as_ref().map(|error| {
            div()
                .p_3()
                .bg(gpui::rgb(0xfef2f2))
                .border_1()
                .border_color(gpui::rgb(0xfecaca))
                .rounded_md()
                .child(
                    div()
                        .text_sm()
                        .text_color(gpui::rgb(0xdc2626))
                        .child(error.clone())
                )
        })
    }
}

impl Render for AddDownloadModal {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let _is_form_valid = self.is_form_valid();
        
        // Get current URL text for submission
        let url_text = self.url_input.read(cx).value().to_string();
        
        Modal::new(window, cx)
            .title("Add Download")
            .width(px(500.0))
            .confirm()
            .button_props(
                gpui_component::modal::ModalButtonProps::default()
                    .ok_text(if self.is_submitting { "Adding..." } else { "Add Download" })
                    .cancel_text("Cancel")
            )
            .on_ok({
                let view_handle = cx.entity();
                let url_text = url_text.clone();
                move |_, _window, cx| {
                    view_handle.update(cx, |this, cx| {
                        if this.is_form_valid() && !this.is_submitting {
                            this.handle_submit(url_text.clone(), cx);
                        }
                    });
                    false // Don't auto-close, let the submit handler decide
                }
            })
            .on_cancel({
                let view_handle = cx.entity();
                move |_, _window, cx| {
                    view_handle.update(cx, |this, cx| {
                        this.handle_cancel(cx);
                    });
                    true // Close modal after handling
                }
            })
            .child(
                div()
                    .v_flex()
                    .gap_4()
                    .p_4()
                    .child(self.render_url_input(cx))
                    .child(self.render_destination_section(cx))
                    .child(self.render_advanced_options(cx))
                    .when_some(self.render_error_message(), |this, error| {
                        this.child(error)
                    })
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_supported_protocol_matching() {
        // Test HTTP
        let http_url = Url::parse("http://example.com/file.zip").unwrap();
        assert!(SupportedProtocol::Http.matches_url(&http_url));
        assert!(!SupportedProtocol::Https.matches_url(&http_url));
        
        // Test HTTPS
        let https_url = Url::parse("https://example.com/file.zip").unwrap();
        assert!(SupportedProtocol::Https.matches_url(&https_url));
        assert!(!SupportedProtocol::Http.matches_url(&https_url));
        
        // Test FTP
        let ftp_url = Url::parse("ftp://example.com/file.zip").unwrap();
        assert!(SupportedProtocol::Ftp.matches_url(&ftp_url));
        
        // Test SFTP
        let sftp_url = Url::parse("sftp://example.com/file.zip").unwrap();
        assert!(SupportedProtocol::Sftp.matches_url(&sftp_url));
        
        // Test magnet link
        let magnet_url = Url::parse("magnet:?xt=urn:btih:example").unwrap();
        assert!(SupportedProtocol::Torrent.matches_url(&magnet_url));
        
        // Test .torrent file URL
        let torrent_file_url = Url::parse("https://example.com/file.torrent").unwrap();
        assert!(SupportedProtocol::Torrent.matches_url(&torrent_file_url));
    }
    
    #[test]
    fn test_url_validation() {
        // Test valid URLs
        assert!(matches!(
            validate_url_string("https://example.com/file.zip"),
            UrlValidationResult::Valid(_, SupportedProtocol::Https)
        ));
        
        assert!(matches!(
            validate_url_string("ftp://example.com/file.zip"),
            UrlValidationResult::Valid(_, SupportedProtocol::Ftp)
        ));
        
        assert!(matches!(
            validate_url_string("magnet:?xt=urn:btih:example"),
            UrlValidationResult::Valid(_, SupportedProtocol::Torrent)
        ));
        
        // Test invalid URLs
        assert!(matches!(
            validate_url_string("not-a-url"),
            UrlValidationResult::Invalid(_)
        ));
        
        assert!(matches!(
            validate_url_string(""),
            UrlValidationResult::Empty
        ));
        
        // Test unsupported protocol
        assert!(matches!(
            validate_url_string("gopher://example.com/file"),
            UrlValidationResult::Invalid(_)
        ));
    }
    
    /// Helper function for testing URL validation
    fn validate_url_string(url_text: &str) -> UrlValidationResult {
        if url_text.trim().is_empty() {
            return UrlValidationResult::Empty;
        }
        
        match Url::parse(url_text.trim()) {
            Ok(url) => {
                let supported_protocols = SupportedProtocol::all();
                
                for protocol in supported_protocols {
                    if protocol.matches_url(&url) {
                        return UrlValidationResult::Valid(url, protocol);
                    }
                }
                
                UrlValidationResult::Invalid(
                    format!("Unsupported protocol '{}'", url.scheme())
                )
            }
            Err(e) => UrlValidationResult::Invalid(format!("Invalid URL: {}", e)),
        }
    }
}