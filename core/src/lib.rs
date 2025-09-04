pub mod api;
pub mod bandwidth;
pub mod callbacks;
pub mod checksum;
pub mod config;
pub mod download;
pub mod engine;
pub mod error;
pub mod event;
pub mod media;
pub mod metalink;
pub mod metrics;
pub mod protocol;
pub mod session;
pub mod types;

// Re-export the high-level API
pub use api::{DownloadResult, Zuup, ZuupBuilder};

pub use bandwidth::{BandwidthLimit, BandwidthManager, BandwidthStats, DownloadBandwidthTracker};
