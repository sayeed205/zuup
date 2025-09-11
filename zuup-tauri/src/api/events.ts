import { listen, type UnlistenFn } from '@tauri-apps/api/event';
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
  static async onDownloadProgress(
    callback: (event: DownloadProgressEvent) => void
  ): Promise<UnlistenFn> {
    return listen('download-progress', (event) => {
      callback(event.payload as DownloadProgressEvent);
    });
  }

  static async onDownloadStateChange(
    callback: (event: DownloadStateChangeEvent) => void
  ): Promise<UnlistenFn> {
    return listen('download-state-change', (event) => {
      callback(event.payload as DownloadStateChangeEvent);
    });
  }

  static async onDownloadAdded(
    callback: (event: DownloadAddedEvent) => void
  ): Promise<UnlistenFn> {
    return listen('download-added', (event) => {
      callback(event.payload as DownloadAddedEvent);
    });
  }

  static async onDownloadRemoved(
    callback: (event: DownloadRemovedEvent) => void
  ): Promise<UnlistenFn> {
    return listen('download-removed', (event) => {
      callback(event.payload as DownloadRemovedEvent);
    });
  }

  static async onStatsUpdate(
    callback: (event: StatsUpdateEvent) => void
  ): Promise<UnlistenFn> {
    return listen('stats-update', (event) => {
      callback(event.payload as StatsUpdateEvent);
    });
  }
}