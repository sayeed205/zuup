use tauri::State;
use crate::services::{DownloadService, download::{AddDownloadRequest, DownloadInfo, DownloadStats}};

#[tauri::command]
pub async fn add_download(
    download_service: State<'_, DownloadService>,
    url: String,
    download_path: Option<String>,
    filename: Option<String>,
) -> Result<String, String> {
    let request = AddDownloadRequest {
        url,
        download_path,
        filename,
    };
    
    download_service.add_download(request)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn pause_download(
    download_service: State<'_, DownloadService>,
    id: String,
) -> Result<(), String> {
    download_service.pause_download(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn resume_download(
    download_service: State<'_, DownloadService>,
    id: String,
) -> Result<(), String> {
    download_service.resume_download(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cancel_download(
    download_service: State<'_, DownloadService>,
    id: String,
) -> Result<(), String> {
    download_service.cancel_download(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn remove_download(
    download_service: State<'_, DownloadService>,
    id: String,
) -> Result<(), String> {
    download_service.remove_download(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_downloads(
    download_service: State<'_, DownloadService>,
) -> Result<Vec<DownloadInfo>, String> {
    Ok(download_service.get_downloads())
}

#[tauri::command]
pub async fn get_download_stats(
    download_service: State<'_, DownloadService>,
) -> Result<DownloadStats, String> {
    Ok(download_service.get_download_stats())
}