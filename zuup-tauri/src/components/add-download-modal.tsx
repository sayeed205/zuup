import { useState } from 'react';
import { DownloadAPI } from '@/api/download';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { X, Download, FolderOpen } from 'lucide-react';

interface AddDownloadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDownloadAdded: () => void;
}

export function AddDownloadModal({ isOpen, onClose, onDownloadAdded }: AddDownloadModalProps) {
  const [url, setUrl] = useState('');
  const [filename, setFilename] = useState('');
  const [downloadPath, setDownloadPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError('URL is required');
      return;
    }

    // Basic URL validation
    try {
      new URL(url);
    } catch {
      setError('Please enter a valid URL');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await DownloadAPI.addDownload(url, {
        filename: filename.trim() || undefined,
        download_path: downloadPath.trim() || undefined,
      });

      // Reset form
      setUrl('');
      setFilename('');
      setDownloadPath('');
      
      onDownloadAdded();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add download');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    if (!isLoading) {
      setUrl('');
      setFilename('');
      setDownloadPath('');
      setError(null);
      onClose();
    }
  };

  const handleBrowseFolder = async () => {
    // TODO: Implement folder selection using Tauri dialog
    // For now, just show a placeholder
    setDownloadPath('Select download folder...');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-lg font-semibold flex items-center">
            <Download className="h-5 w-5 mr-2" />
            Add Download
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClose}
            disabled={isLoading}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* URL Input */}
          <div>
            <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-1">
              URL *
            </label>
            <Input
              id="url"
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/file.zip"
              disabled={isLoading}
              className="w-full"
            />
          </div>

          {/* Filename Input */}
          <div>
            <label htmlFor="filename" className="block text-sm font-medium text-gray-700 mb-1">
              Filename (optional)
            </label>
            <Input
              id="filename"
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="Leave empty to auto-detect"
              disabled={isLoading}
              className="w-full"
            />
          </div>

          {/* Download Path */}
          <div>
            <label htmlFor="downloadPath" className="block text-sm font-medium text-gray-700 mb-1">
              Download Location (optional)
            </label>
            <div className="flex space-x-2">
              <Input
                id="downloadPath"
                type="text"
                value={downloadPath}
                onChange={(e) => setDownloadPath(e.target.value)}
                placeholder="Leave empty for default location"
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                onClick={handleBrowseFolder}
                disabled={isLoading}
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || !url.trim()}
              className="flex items-center space-x-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Adding...</span>
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" />
                  <span>Add Download</span>
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
