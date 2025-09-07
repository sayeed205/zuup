//! Protocol handlers module with conditional compilation support

// Conditional module declarations based on features
#[cfg(feature = "http")]
pub mod http;

#[cfg(feature = "ftp")]
pub mod ftp;

#[cfg(feature = "sftp")]
pub mod sftp;

#[cfg(feature = "torrent")]
pub mod torrent;

// Conditional re-exports for protocol handlers
#[cfg(feature = "http")]
pub use http::HttpProtocolHandler;

#[cfg(feature = "ftp")]
pub use ftp::FtpProtocolHandler;

#[cfg(feature = "sftp")]
pub use sftp::SftpProtocolHandler;

#[cfg(feature = "torrent")]
pub use torrent::BitTorrentProtocolHandler;