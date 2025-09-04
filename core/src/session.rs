use std::sync::Arc;

pub struct SessionManager {
    /// Session storage backend
    storage: Arc<SessionStorage>,
}

/// Session storage backend using sled embedded database - todo))
pub struct SessionStorage {
    // Sled database instance
    // db: sled::Db,
}
