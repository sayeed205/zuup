//! Theme system and system integration for the GUI application

use std::sync::Arc;
use gpui::*;
use gpui_component::theme::{Theme, ThemeMode as ComponentThemeMode};
use crate::config::{ThemeMode, ConfigManager};
use std::sync::Mutex;
use dark_light::Mode as SystemMode;

/// Theme manager handles theme switching and system integration
pub struct ThemeManager {
    current_mode: ThemeMode,
    config_manager: Arc<Mutex<ConfigManager>>,
    system_theme_watcher: Option<SystemThemeWatcher>,
}

impl ThemeManager {
    /// Create a new theme manager
    pub fn new(config_manager: Arc<Mutex<ConfigManager>>) -> Self {
        let current_mode = {
            let config = config_manager.lock().unwrap();
            config.gui_config().theme
        };

        Self {
            current_mode,
            config_manager,
            system_theme_watcher: None,
        }
    }

    /// Initialize the theme manager and start system theme watching if needed
    pub fn initialize(&mut self, cx: &mut App) -> anyhow::Result<()> {
        // Start system theme watcher if using system theme
        if self.current_mode == ThemeMode::System {
            self.start_system_theme_watcher(cx)?;
        }

        // Apply the current theme
        self.apply_theme(cx)?;

        Ok(())
    }

    /// Get the current theme mode
    pub fn current_mode(&self) -> ThemeMode {
        self.current_mode
    }

    /// Set the theme mode and apply it
    pub fn set_theme_mode(&mut self, mode: ThemeMode, cx: &mut App) -> anyhow::Result<()> {
        if self.current_mode == mode {
            return Ok(());
        }

        self.current_mode = mode;

        // Update configuration
        {
            let mut config = self.config_manager.lock().unwrap();
            config.gui_config_mut().theme = mode;
            config.save_gui_config().map_err(|e| anyhow::anyhow!("Failed to save config: {}", e))?;
        }

        // Handle system theme watcher
        match mode {
            ThemeMode::System => {
                if self.system_theme_watcher.is_none() {
                    self.start_system_theme_watcher(cx)?;
                }
            }
            ThemeMode::Light | ThemeMode::Dark => {
                self.stop_system_theme_watcher();
            }
        }

        // Apply the new theme
        self.apply_theme(cx)?;

        Ok(())
    }

    /// Apply the current theme to the application
    fn apply_theme(&self, cx: &mut App) -> anyhow::Result<()> {
        let effective_mode = self.get_effective_theme_mode();
        let component_theme_mode = match effective_mode {
            ThemeMode::Light => ComponentThemeMode::Light,
            ThemeMode::Dark => ComponentThemeMode::Dark,
            ThemeMode::System => {
                // This shouldn't happen as get_effective_theme_mode resolves System
                ComponentThemeMode::Light
            }
        };

        // Apply theme to gpui-component using the Theme::change method
        Theme::change(component_theme_mode, None, cx);

        Ok(())
    }

    /// Get the effective theme mode (resolves System to Light/Dark)
    fn get_effective_theme_mode(&self) -> ThemeMode {
        match self.current_mode {
            ThemeMode::System => self.detect_system_theme(),
            mode => mode,
        }
    }

    /// Detect the current system theme
    fn detect_system_theme(&self) -> ThemeMode {
        SystemThemeDetector::detect_system_theme()
    }

    /// Start watching for system theme changes
    fn start_system_theme_watcher(&mut self, cx: &mut App) -> anyhow::Result<()> {
        if self.system_theme_watcher.is_some() {
            return Ok(());
        }

        let watcher = SystemThemeWatcher::new(cx)?;
        self.system_theme_watcher = Some(watcher);

        Ok(())
    }

    /// Stop watching for system theme changes
    fn stop_system_theme_watcher(&mut self) {
        self.system_theme_watcher = None;
    }

    /// Handle system theme change notification
    pub fn on_system_theme_changed(&mut self, cx: &mut App) -> anyhow::Result<()> {
        if self.current_mode == ThemeMode::System {
            self.apply_theme(cx)?;
        }
        Ok(())
    }
}

/// System theme detector using the dark-light crate
pub struct SystemThemeDetector;

impl SystemThemeDetector {
    /// Detect the current system theme using dark-light crate
    pub fn detect_system_theme() -> ThemeMode {
        match dark_light::detect() {
            Ok(SystemMode::Dark) => ThemeMode::Dark,
            Ok(SystemMode::Light) => ThemeMode::Light,
            Ok(SystemMode::Unspecified) => ThemeMode::Light, // Default to light for unspecified
            Err(_) => ThemeMode::Light, // Default to light on error
        }
    }
}

/// System theme watcher for monitoring theme changes
pub struct SystemThemeWatcher {
    _handle: Option<Box<dyn std::any::Any + Send>>,
}

impl SystemThemeWatcher {
    /// Create a new system theme watcher
    pub fn new(_cx: &mut App) -> anyhow::Result<Self> {
        // For now, we'll create a simple watcher that doesn't actively monitor
        // In a full implementation, this would set up platform-specific watchers
        Ok(Self {
            _handle: None,
        })
    }
}

