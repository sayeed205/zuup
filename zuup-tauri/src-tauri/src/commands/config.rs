use crate::services::{
    ConfigService,
    config::{AppConfig, GuiConfig},
};
use tauri::State;

#[tauri::command]
pub async fn load_config(config_service: State<'_, ConfigService>) -> Result<AppConfig, String> {
    Ok(config_service.get_app_config())
}

#[tauri::command]
pub async fn save_config(
    config_service: State<'_, ConfigService>,
    config: AppConfig,
) -> Result<(), String> {
    config_service
        .update_app_config(config)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_gui_config(config_service: State<'_, ConfigService>) -> Result<GuiConfig, String> {
    Ok(config_service.get_gui_config())
}

#[tauri::command]
pub async fn update_gui_config(
    config_service: State<'_, ConfigService>,
    config: GuiConfig,
) -> Result<(), String> {
    config_service
        .update_gui_config(config)
        .map_err(|e| e.to_string())
}
