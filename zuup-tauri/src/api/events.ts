import { listen } from '@tauri-apps/api/event';
import type { UnlistenFn } from '@tauri-apps/api/event';
import type { DownloadInfo, DownloadStats, ProgressInfo, DownloadState } from '@/types';

export interface DownloadProgressEvent {
  id: string;
  progress: ProgressInfo;
}

export interface DownloadStateChangeEvent {
  id: string;
  state: DownloadState;
}

export interface DownloadAddedEvent {
  download: DownloadInfo;
}

export interface DownloadRemovedEvent {
  id: string;
}

export interface StatsUpdateEvent {
  stats: DownloadStats;
}

export class EventAPI {
  static onDownloadProgress(callback: (event: DownloadProgressEvent) => void): Promise<UnlistenFn> {
    return listen('download-progress', (event) => {
      callback(event.payload as DownloadProgressEvent);
    });
  }

  static onDownloadStateChange(callback: (event: DownloadStateChangeEvent) => void): Promise<UnlistenFn> {
    return listen('download-state-change', (event) => {
      callback(event.payload as DownloadStateChangeEvent);
    });
  }

  static onDownloadAdded(callback: (event: DownloadAddedEvent) => void): Promise<UnlistenFn> {
    return listen('download-added', (event) => {
      callback(event.payload as DownloadAddedEvent);
    });
  }

  static onDownloadRemoved(callback: (event: DownloadRemovedEvent) => void): Promise<UnlistenFn> {
    return listen('download-removed', (event) => {
      callback(event.payload as DownloadRemovedEvent);
    });
  }

  static onStatsUpdate(callback: (event: StatsUpdateEvent) => void): Promise<UnlistenFn> {
    return listen('stats-update', (event) => {
      callback(event.payload as StatsUpdateEvent);
    });
  }
}