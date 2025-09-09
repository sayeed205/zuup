//! UI views and components for the download manager

pub mod download_manager;
pub mod download_item;
pub mod add_download_modal;
pub mod confirmation_dialog;

pub use download_manager::DownloadManagerView;
pub use download_item::DownloadItemView;
pub use add_download_modal::AddDownloadModal;
pub use confirmation_dialog::{ConfirmationDialog, create_remove_confirmation_dialog, create_cancel_confirmation_dialog};