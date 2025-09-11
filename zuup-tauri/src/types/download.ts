export interface DownloadInfo {
  id: string;
  url: string;
  filename: string;
  file_size?: number;
  downloaded_size: number;
  progress: ProgressInfo;
  state: DownloadState;
  created_at: string;
  completed_at?: string;
  error?: string;
}

export interface ProgressInfo {
  percentage: number;
  download_speed: number;
  upload_speed: number;
  eta?: number;
  connections: number;
}

export type DownloadState = 
  | 'preparing'
  | 'active'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface DownloadStats {
  total_downloads: number;
  active_downloads: number;
  completed_downloads: number;
  failed_downloads: number;
  paused_downloads: number;
  total_download_speed: number;
  total_upload_speed: number;
  overall_progress: number;
  eta?: number;
}

export interface AddDownloadRequest {
  url: string;
  download_path?: string;
  filename?: string;
}