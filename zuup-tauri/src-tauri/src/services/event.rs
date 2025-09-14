use super::download::{DownloadInfo, DownloadState, DownloadStats, ProgressInfo};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
pub struct DownloadProgressEvent {
    pub id: String,
    pub progress: ProgressInfo,
}

#[derive(Clone, Serialize)]
pub struct DownloadStateChangeEvent {
    pub id: String,
    pub state: DownloadState,
}

#[derive(Clone, Serialize)]
pub struct DownloadAddedEvent {
    pub download: DownloadInfo,
}

#[derive(Clone, Serialize)]
pub struct DownloadRemovedEvent {
    pub id: String,
}

#[derive(Clone, Serialize)]
pub struct StatsUpdateEvent {
    pub stats: DownloadStats,
}

#[derive(Clone)]
pub struct EventService {
    app_handle: AppHandle,
}

impl EventService {
    pub fn new(app_handle: AppHandle) -> Self {
        Self { app_handle }
    }

    pub fn emit_download_progress(&self, id: String, progress: ProgressInfo) {
        let event = DownloadProgressEvent { id, progress };
        if let Err(e) = self.app_handle.emit("download-progress", event) {
            tracing::error!("Failed to emit download progress event: {}", e);
        }
    }

    pub fn emit_download_state_change(&self, id: String, state: DownloadState) {
        let event = DownloadStateChangeEvent { id, state };
        if let Err(e) = self.app_handle.emit("download-state-change", event) {
            tracing::error!("Failed to emit download state change event: {}", e);
        }
    }

    pub fn emit_download_added(&self, download: DownloadInfo) {
        let event = DownloadAddedEvent { download };
        if let Err(e) = self.app_handle.emit("download-added", event) {
            tracing::error!("Failed to emit download added event: {}", e);
        }
    }

    pub fn emit_download_removed(&self, id: String) {
        let event = DownloadRemovedEvent { id };
        if let Err(e) = self.app_handle.emit("download-removed", event) {
            tracing::error!("Failed to emit download removed event: {}", e);
        }
    }

    pub fn emit_stats_update(&self, stats: DownloadStats) {
        let event = StatsUpdateEvent { stats };
        if let Err(e) = self.app_handle.emit("stats-update", event) {
            tracing::error!("Failed to emit stats update event: {}", e);
        }
    }
}
