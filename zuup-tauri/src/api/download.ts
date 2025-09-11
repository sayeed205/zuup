import { invoke } from '@tauri-apps/api/core';
import type { DownloadInfo, DownloadStats } from '@/types';

export class DownloadAPI {
  static async addDownload(
    url: string, 
    options?: { download_path?: string; filename?: string }
  ): Promise<string> {
    return invoke('add_download', {
      url,
      downloadPath: options?.download_path,
      filename: options?.filename,
    });
  }

  static async pauseDownload(id: string): Promise<void> {
    return invoke('pause_download', { id });
  }

  static async resumeDownload(id: string): Promise<void> {
    return invoke('resume_download', { id });
  }

  static async cancelDownload(id: string): Promise<void> {
    return invoke('cancel_download', { id });
  }

  static async removeDownload(id: string): Promise<void> {
    return invoke('remove_download', { id });
  }

  static async getDownloads(): Promise<DownloadInfo[]> {
    return invoke('get_downloads');
  }

  static async getDownloadStats(): Promise<DownloadStats> {
    return invoke('get_download_stats');
  }
}