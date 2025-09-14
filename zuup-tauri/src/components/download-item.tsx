import { useState } from 'react';
import type { DownloadInfo } from '@/types';
import { DownloadAPI } from '@/api/download';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Pause, 
  X, 
  Trash2, 
  MoreVertical,
  File,
  Download,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface DownloadItemProps {
  download: DownloadInfo;
  onUpdate: () => void;
}

export function DownloadItem({ download, onUpdate }: DownloadItemProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleAction = async (action: () => Promise<void>) => {
    try {
      setIsLoading(true);
      await action();
      onUpdate();
    } catch (error) {
      console.error('Action failed:', error);
      // TODO: Show error toast
    } finally {
      setIsLoading(false);
    }
  };

  const handlePause = () => handleAction(() => DownloadAPI.pauseDownload(download.id));
  const handleResume = () => handleAction(() => DownloadAPI.resumeDownload(download.id));
  const handleCancel = () => handleAction(() => DownloadAPI.cancelDownload(download.id));
  const handleRemove = () => handleAction(() => DownloadAPI.removeDownload(download.id));

  const getStateIcon = () => {
    switch (download.state) {
      case 'active':
        return <Download className="h-4 w-4 text-blue-600" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-600" />;
      case 'preparing':
        return <Clock className="h-4 w-4 text-gray-600" />;
      default:
        return <File className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStateColor = () => {
    switch (download.state) {
      case 'active':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'preparing':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatSpeed = (bytesPerSecond: number): string => {
    return formatBytes(bytesPerSecond) + '/s';
  };

  const formatETA = (eta?: number): string => {
    if (!eta) return 'Unknown';
    const hours = Math.floor(eta / 3600);
    const minutes = Math.floor((eta % 3600) / 60);
    const seconds = eta % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const getProgressColor = () => {
    if (download.state === 'failed') return 'bg-red-500';
    if (download.state === 'completed') return 'bg-green-500';
    if (download.state === 'active') return 'bg-blue-500';
    return 'bg-gray-500';
  };

  return (
    <div className="p-4 hover:bg-gray-50 transition-colors">
      <div className="flex items-start space-x-4">
        {/* Status Icon */}
        <div className="flex-shrink-0 mt-1">
          {getStateIcon()}
        </div>

        {/* Main Content */}
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 min-w-0">
              <h3 className="text-sm font-medium text-gray-900 truncate">
                {download.filename}
              </h3>
              <Badge className={`text-xs ${getStateColor()}`}>
                {download.state}
              </Badge>
            </div>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" disabled={isLoading}>
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {download.state === 'active' && (
                  <DropdownMenuItem onClick={handlePause} disabled={isLoading}>
                    <Pause className="h-4 w-4 mr-2" />
                    Pause
                  </DropdownMenuItem>
                )}
                {download.state === 'paused' && (
                  <DropdownMenuItem onClick={handleResume} disabled={isLoading}>
                    <Play className="h-4 w-4 mr-2" />
                    Resume
                  </DropdownMenuItem>
                )}
                {(download.state === 'active' || download.state === 'paused') && (
                  <DropdownMenuItem onClick={handleCancel} disabled={isLoading}>
                    <X className="h-4 w-4 mr-2" />
                    Cancel
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={handleRemove} disabled={isLoading}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Remove
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* URL */}
          <p className="text-xs text-gray-500 truncate mt-1">
            {download.url}
          </p>

          {/* Progress Bar */}
          {download.state !== 'completed' && download.state !== 'cancelled' && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>{download.progress.percentage.toFixed(1)}%</span>
                <span>
                  {formatBytes(download.downloaded_size)}
                  {download.file_size && ` / ${formatBytes(download.file_size)}`}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${getProgressColor()}`}
                  style={{ width: `${Math.min(download.progress.percentage, 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Stats */}
          <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
            <div className="flex space-x-4">
              {download.state === 'active' && (
                <>
                  <span>Speed: {formatSpeed(download.progress.download_speed)}</span>
                  {download.progress.eta && (
                    <span>ETA: {formatETA(download.progress.eta)}</span>
                  )}
                  <span>Connections: {download.progress.connections}</span>
                </>
              )}
              {download.state === 'completed' && (
                <span>Completed</span>
              )}
              {download.state === 'failed' && (
                <span className="text-red-600">Failed</span>
              )}
              {download.state === 'paused' && (
                <span>Paused</span>
              )}
            </div>
            <span>
              {new Date(download.created_at).toLocaleDateString()}
            </span>
          </div>

          {/* Error Message */}
          {download.error && (
            <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
              {download.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
