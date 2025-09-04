use cuid2::cuid;
use serde::{Deserialize, Serialize};

/// Unique identifier for downloads
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloadId(String);

impl DownloadId {
    /// Create a new unique download ID
    pub fn new() -> Self {
        Self(cuid())
    }

    /// Create a download ID from a string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Get the string representation of the ID
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DownloadId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DownloadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
