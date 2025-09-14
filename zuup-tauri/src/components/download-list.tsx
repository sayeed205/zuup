import { useState, useEffect } from 'react';
import { DownloadAPI, EventAPI } from '@/api';
import type { DownloadInfo, DownloadStats } from '@/types';
import { DownloadItem } from './download-item';
import { Button } from '@/components/ui/button';
import { Plus, Filter, SortAsc } from 'lucide-react';

interface DownloadListProps {
  onAddDownload: () => void;
}

export function DownloadList({ onAddDownload }: DownloadListProps) {
  const [downloads, setDownloads] = useState<DownloadInfo[]>([]);
  const [stats, setStats] = useState<DownloadStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'active' | 'completed' | 'failed' | 'paused'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'size' | 'progress' | 'speed' | 'date'>('date');

  const loadDownloads = async () => {
    try {
      setLoading(true);
      const [downloadsData, statsData] = await Promise.all([
        DownloadAPI.getDownloads(),
        DownloadAPI.getDownloadStats()
      ]);
      setDownloads(downloadsData);
      setStats(statsData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load downloads');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDownloads();
    
    // Set up real-time event listeners
    const setupEventListeners = async () => {
      const unlisteners = await Promise.all([
        EventAPI.onDownloadProgress((event) => {
          setDownloads(prev => prev.map(download => 
            download.id === event.id 
              ? { ...download, progress: event.progress }
              : download
          ));
        }),
        EventAPI.onDownloadStateChange((event) => {
          setDownloads(prev => prev.map(download => 
            download.id === event.id 
              ? { ...download, state: event.state }
              : download
          ));
        }),
        EventAPI.onDownloadAdded((event) => {
          setDownloads(prev => [event.download, ...prev]);
        }),
        EventAPI.onDownloadRemoved((event) => {
          setDownloads(prev => prev.filter(download => download.id !== event.id));
        }),
        EventAPI.onStatsUpdate((event) => {
          setStats(event.stats);
        }),
      ]);

      return () => {
        unlisteners.forEach(unlisten => unlisten());
      };
    };

    setupEventListeners();
  }, []);

  const filteredDownloads = downloads.filter(download => {
    switch (filter) {
      case 'active':
        return download.state === 'active';
      case 'completed':
        return download.state === 'completed';
      case 'failed':
        return download.state === 'failed';
      case 'paused':
        return download.state === 'paused';
      default:
        return true;
    }
  });

  const sortedDownloads = [...filteredDownloads].sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.filename.localeCompare(b.filename);
      case 'size':
        return (b.file_size || 0) - (a.file_size || 0);
      case 'progress':
        return b.progress.percentage - a.progress.percentage;
      case 'speed':
        return b.progress.download_speed - a.progress.download_speed;
      case 'date':
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      default:
        return 0;
    }
  });

  const getFilterCount = (filterType: typeof filter) => {
    if (!stats) return 0;
    switch (filterType) {
      case 'active':
        return stats.active_downloads;
      case 'completed':
        return stats.completed_downloads;
      case 'failed':
        return stats.failed_downloads;
      case 'paused':
        return stats.paused_downloads;
      default:
        return stats.total_downloads;
    }
  };

  if (loading && downloads.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading downloads...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-semibold">Downloads</h2>
          {stats && (
            <div className="text-sm text-gray-600">
              {getFilterCount(filter)} of {stats.total_downloads} downloads
            </div>
          )}
        </div>
        <Button onClick={onAddDownload} className="flex items-center space-x-2">
          <Plus className="h-4 w-4" />
          <span>Add Download</span>
        </Button>
      </div>

      {/* Filters and Controls */}
      <div className="flex items-center justify-between p-4 border-b bg-gray-50">
        <div className="flex space-x-1">
          {(['all', 'active', 'completed', 'failed', 'paused'] as const).map((filterType) => (
            <Button
              key={filterType}
              variant={filter === filterType ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilter(filterType)}
              className="capitalize"
            >
              {filterType} ({getFilterCount(filterType)})
            </Button>
          ))}
        </div>
        
        <div className="flex items-center space-x-2">
          <SortAsc className="h-4 w-4 text-gray-500" />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            className="text-sm border rounded px-2 py-1"
          >
            <option value="date">Date Added</option>
            <option value="name">Name</option>
            <option value="size">Size</option>
            <option value="progress">Progress</option>
            <option value="speed">Speed</option>
          </select>
        </div>
      </div>

      {/* Download List */}
      <div className="flex-1 overflow-y-auto">
        {error && (
          <div className="p-4 bg-red-50 border-l-4 border-red-400">
            <p className="text-red-700">{error}</p>
            <Button 
              onClick={loadDownloads} 
              variant="outline" 
              size="sm" 
              className="mt-2"
            >
              Retry
            </Button>
          </div>
        )}

        {sortedDownloads.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <Filter className="h-12 w-12 mb-4 opacity-50" />
            <p className="text-lg font-medium">No downloads found</p>
            <p className="text-sm">
              {filter === 'all' 
                ? 'Add a download to get started' 
                : `No ${filter} downloads found`
              }
            </p>
          </div>
        ) : (
          <div className="divide-y">
            {sortedDownloads.map((download) => (
              <DownloadItem
                key={download.id}
                download={download}
                onUpdate={loadDownloads}
              />
            ))}
          </div>
        )}
      </div>

      {/* Status Bar */}
      {stats && (
        <div className="p-4 border-t bg-gray-50">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex space-x-4">
              <span>Active: {stats.active_downloads}</span>
              <span>Completed: {stats.completed_downloads}</span>
              <span>Failed: {stats.failed_downloads}</span>
              <span>Paused: {stats.paused_downloads}</span>
            </div>
            <div className="flex space-x-4">
              <span>Speed: {formatBytes(stats.total_download_speed)}/s</span>
              <span>Progress: {stats.overall_progress.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
