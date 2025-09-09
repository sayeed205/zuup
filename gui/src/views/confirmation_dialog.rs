//! Confirmation dialog component for user confirmations

use std::sync::Arc;
use gpui::*;
use gpui::prelude::FluentBuilder;
use gpui_component::{
    button::Button,
    StyledExt,
};

/// Confirmation dialog for user actions
pub struct ConfirmationDialog {
    /// Dialog title
    title: String,
    
    /// Dialog message
    message: String,
    
    /// Confirm button text
    confirm_text: String,
    
    /// Cancel button text
    cancel_text: String,
    
    /// Whether to show file deletion option
    show_delete_file_option: bool,
    
    /// Whether file deletion is selected
    delete_file_selected: bool,
    
    /// Callback for confirm action
    on_confirm: Option<Arc<dyn Fn(bool) + Send + Sync>>,
    
    /// Callback for cancel action
    on_cancel: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl ConfirmationDialog {
    /// Create a new confirmation dialog
    pub fn new(title: String, message: String) -> Self {
        Self {
            title,
            message,
            confirm_text: "Confirm".to_string(),
            cancel_text: "Cancel".to_string(),
            show_delete_file_option: false,
            delete_file_selected: false,
            on_confirm: None,
            on_cancel: None,
        }
    }
    
    /// Set the confirm button text
    pub fn confirm_text(mut self, text: String) -> Self {
        self.confirm_text = text;
        self
    }
    
    /// Set the cancel button text
    pub fn cancel_text(mut self, text: String) -> Self {
        self.cancel_text = text;
        self
    }
    
    /// Show file deletion option
    pub fn with_delete_file_option(mut self) -> Self {
        self.show_delete_file_option = true;
        self
    }
    
    /// Set the confirm callback
    pub fn on_confirm<F>(mut self, callback: F) -> Self
    where
        F: Fn(bool) + Send + Sync + 'static,
    {
        self.on_confirm = Some(Arc::new(callback));
        self
    }
    
    /// Set the cancel callback
    pub fn on_cancel<F>(mut self, callback: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_cancel = Some(Arc::new(callback));
        self
    }
    
    /// Handle confirm action
    fn handle_confirm(&mut self, _cx: &mut Context<Self>) {
        if let Some(callback) = &self.on_confirm {
            callback(self.delete_file_selected);
        }
    }
    
    /// Handle cancel action
    fn handle_cancel(&mut self, _cx: &mut Context<Self>) {
        if let Some(callback) = &self.on_cancel {
            callback();
        }
    }
    
    /// Toggle file deletion option
    fn toggle_delete_file(&mut self, _cx: &mut Context<Self>) {
        self.delete_file_selected = !self.delete_file_selected;
    }
}

impl Render for ConfirmationDialog {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Modal overlay
        div()
            .absolute()
            .inset_0()
            .bg(gpui::rgba(0x000000aa)) // Semi-transparent black overlay
            .flex()
            .items_center()
            .justify_center()
            .child(
                // Dialog box
                div()
                    .bg(gpui::rgb(0xffffff))
                    .rounded_lg()
                    .shadow_lg()
                    .p_6()
                    .min_w(px(400.0))
                    .max_w(px(500.0))
                    .child(
                        div()
                            .v_flex()
                            .gap_4()
                            .child(
                                // Title
                                div()
                                    .text_lg()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(gpui::rgb(0x111827))
                                    .child(self.title.clone())
                            )
                            .child(
                                // Message
                                div()
                                    .text_sm()
                                    .text_color(gpui::rgb(0x6b7280))
                                    .child(self.message.clone())
                            )
                            .when(self.show_delete_file_option, |this| {
                                this.child(
                                    // File deletion option
                                    div()
                                        .h_flex()
                                        .items_center()
                                        .gap_2()
                                        .p_2()
                                        .bg(gpui::rgb(0xf9fafb))
                                        .rounded_sm()
                                        .border_1()
                                        .border_color(gpui::rgb(0xd1d5db))
                                        .child(
                                            Button::new("delete_file_checkbox")
                                                .label(if self.delete_file_selected { "☑" } else { "☐" })
                                                .on_click(cx.listener(|this, _event, _window, cx| {
                                                    this.toggle_delete_file(cx);
                                                }))
                                        )
                                        .child(
                                            div()
                                                .text_sm()
                                                .text_color(gpui::rgb(0x374151))
                                                .child("Also delete the downloaded file from disk")
                                        )
                                )
                            })
                            .child(
                                // Button row
                                div()
                                    .h_flex()
                                    .justify_end()
                                    .gap_3()
                                    .child(
                                        Button::new("cancel")
                                            .label(&self.cancel_text)
                                            .on_click(cx.listener(|this, _event, _window, cx| {
                                                this.handle_cancel(cx);
                                            }))
                                    )
                                    .child(
                                        Button::new("confirm")
                                            .label(&self.confirm_text)
                                            .on_click(cx.listener(|this, _event, _window, cx| {
                                                this.handle_confirm(cx);
                                            }))
                                    )
                            )
                    )
            )
    }
}

/// Create a confirmation dialog for removing a download
pub fn create_remove_confirmation_dialog(
    filename: &str,
    is_completed: bool,
    on_confirm: impl Fn(bool) + Send + Sync + 'static,
    on_cancel: impl Fn() + Send + Sync + 'static,
) -> ConfirmationDialog {
    let title = "Remove Download".to_string();
    let message = format!("Are you sure you want to remove \"{}\" from the download list?", filename);
    
    let mut dialog = ConfirmationDialog::new(title, message)
        .confirm_text("Remove".to_string())
        .cancel_text("Cancel".to_string())
        .on_confirm(on_confirm)
        .on_cancel(on_cancel);
    
    if is_completed {
        dialog = dialog.with_delete_file_option();
    }
    
    dialog
}

/// Create a confirmation dialog for cancelling a download
pub fn create_cancel_confirmation_dialog(
    filename: &str,
    on_confirm: impl Fn(bool) + Send + Sync + 'static,
    on_cancel: impl Fn() + Send + Sync + 'static,
) -> ConfirmationDialog {
    let title = "Cancel Download".to_string();
    let message = format!("Are you sure you want to cancel the download of \"{}\"? This will stop the download and remove it from the list.", filename);
    
    ConfirmationDialog::new(title, message)
        .confirm_text("Cancel Download".to_string())
        .cancel_text("Keep Downloading".to_string())
        .on_confirm(on_confirm)
        .on_cancel(on_cancel)
}