export interface AppConfig {
  general: GeneralConfig;
  network: NetworkConfig;
}

export interface GeneralConfig {
  download_directory: string;
  max_concurrent_downloads: number;
  auto_start_downloads: boolean;
}

export interface NetworkConfig {
  max_connections_per_download: number;
  connection_timeout: number;
  read_timeout: number;
}

export interface GuiConfig {
  theme: 'system' | 'light' | 'dark';
  window_controls_visible: boolean;
  window_size: [number, number];
  window_position?: [number, number];
  sidebar_width: number;
  auto_start_downloads: boolean;
  show_notifications: boolean;
  minimize_to_tray: boolean;
  start_minimized: boolean;
  close_to_tray: boolean;
}