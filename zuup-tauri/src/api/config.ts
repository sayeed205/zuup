import { invoke } from '@tauri-apps/api/core';
import type { AppConfig, GuiConfig } from '@/types';

export class ConfigAPI {
  static async loadConfig(): Promise<AppConfig> {
    return invoke('load_config');
  }

  static async saveConfig(config: AppConfig): Promise<void> {
    return invoke('save_config', { config });
  }

  static async getGuiConfig(): Promise<GuiConfig> {
    return invoke('get_gui_config');
  }

  static async updateGuiConfig(config: GuiConfig): Promise<void> {
    return invoke('update_gui_config', { config });
  }
}