#![recursion_limit = "256"]

use gpui::*;
use gpui_component::Root;
use std::sync::Arc;

mod config;
mod theme;
mod engine;
mod app;
mod views;

use app::ZuupApp;
use config::ThemeMode;
use theme::SystemThemeDetector;
use views::DownloadManagerView;

/// Main application window view
pub struct MainWindowView {
    /// Reference to the main application
    app: Arc<ZuupApp>,
    /// Current theme for display purposes
    current_theme: ThemeMode,
    /// Engine status message
    engine_status: String,
    /// Download manager view
    download_manager: Entity<DownloadManagerView>,
}

impl MainWindowView {
    pub fn new(app: Arc<ZuupApp>, cx: &mut Context<Self>) -> Self {
        let current_theme = {
            let config = app.config_manager().lock().unwrap();
            config.gui_config().theme
        };
        
        let download_manager = cx.new(|_| DownloadManagerView::new(app.clone()));
        
        Self { 
            app,
            current_theme,
            engine_status: "Engine integration ready".to_string(),
            download_manager,
        }
    }
    
    fn get_effective_theme(&self) -> ThemeMode {
        match self.current_theme {
            ThemeMode::System => SystemThemeDetector::detect_system_theme(),
            mode => mode,
        }
    }
    
    /// Handle theme change
    fn handle_theme_change(&mut self, new_theme: ThemeMode, _cx: &mut Context<Self>) {
        self.current_theme = new_theme;
        
        // Update configuration
        {
            let mut config = self.app.config_manager().lock().unwrap();
            config.gui_config_mut().theme = new_theme;
            if let Err(e) = config.save_gui_config() {
                eprintln!("Failed to save theme configuration: {}", e);
            }
        }
        
        // TODO: Update theme manager when we have proper GPUI app context access
        // For now, just update the configuration and the theme will be applied on next restart
        
        tracing::info!("Theme changed to {:?}", new_theme);
    }
}

impl Render for MainWindowView {
    fn render(&mut self, _: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(self.download_manager.clone())
    }
}

fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();
    
    let app = Application::new();

    app.run(move |cx| {
        gpui_component::init(cx);
        cx.activate(true);

        // Initialize the main application asynchronously
        cx.spawn(async move |cx| {
            // Create a simplified Zuup application for now (without full engine integration)
            let zuup_app = match ZuupApp::new_simple() {
                Ok(app) => Arc::new(app),
                Err(e) => {
                    eprintln!("Failed to initialize Zuup application: {}", e);
                    return Err(anyhow::anyhow!("Application initialization failed: {}", e));
                }
            };

            // Get window configuration from the config manager
            let (window_size, window_position, window_controls_visible) = {
                let config = zuup_app.config_manager().lock().unwrap();
                let gui_config = config.gui_config();
                (
                    gui_config.window_size,
                    gui_config.window_position,
                    gui_config.window_controls_visible,
                )
            };

            // Calculate window bounds (simplified without app context for now)
            let window_bounds = calculate_window_bounds(window_size, window_position);

            // Configure window options based on settings
            let window_options = create_window_options(window_bounds, window_controls_visible);

            // Create and open the main window
            let window = cx
                .open_window(window_options, {
                    let zuup_app = zuup_app.clone();
                    move |window, cx| {
                        // Initialize the application with the proper GPUI context
                        if let Err(e) = zuup_app.initialize_sync(cx) {
                            eprintln!("Failed to initialize application context: {}", e);
                        }
                        
                        let main_view = cx.new(|cx| MainWindowView::new(zuup_app.clone(), cx));
                        cx.new(|cx| Root::new(main_view.into(), window, cx))
                    }
                })
                .expect("Failed to open main window");

            // Configure the window
            window
                .update(cx, |_, window, _| {
                    window.activate_window();
                    window.set_window_title("Zuup Download Manager");
                })
                .expect("Failed to configure window");

            // Set up window event handlers for position and size persistence
            setup_window_persistence(window.clone(), zuup_app.clone());

            tracing::info!("Zuup Download Manager started successfully");
            Ok::<_, anyhow::Error>(())
        })
        .detach();
    });
}

/// Calculate window bounds based on configuration and display constraints
fn calculate_window_bounds(
    window_size: (u32, u32),
    window_position: Option<(i32, i32)>,
) -> Bounds<Pixels> {
    let size = size(px(window_size.0 as f32), px(window_size.1 as f32));
    
    // For now, use the configured size directly
    // TODO: Add display bounds checking when we have proper display access
    let constrained_size = size;

    // Use saved position or default position
    match window_position {
        Some((x, y)) => {
            let origin = point(px(x as f32), px(y as f32));
            Bounds::new(origin, constrained_size)
        }
        None => {
            // Default to a reasonable position if no saved position
            let origin = point(px(100.0), px(100.0));
            Bounds::new(origin, constrained_size)
        }
    }
}

/// Create window options based on configuration
fn create_window_options(
    window_bounds: Bounds<Pixels>,
    window_controls_visible: bool,
) -> WindowOptions {
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(window_bounds)),
        titlebar: if window_controls_visible {
            Some(TitlebarOptions {
                title: Some("Zuup Download Manager".into()),
                appears_transparent: false,
                traffic_light_position: None,
            })
        } else {
            None
        },
        window_min_size: Some(Size {
            width: px(640.0),
            height: px(480.0),
        }),
        kind: WindowKind::Normal,
        is_movable: true,
        display_id: None,
        
        #[cfg(target_os = "linux")]
        window_background: if window_controls_visible {
            WindowBackgroundAppearance::Opaque
        } else {
            WindowBackgroundAppearance::Transparent
        },
        
        #[cfg(target_os = "linux")]
        window_decorations: if window_controls_visible {
            Some(WindowDecorations::Server)
        } else {
            Some(WindowDecorations::Client)
        },
        
        ..Default::default()
    }
}

/// Set up window event handlers for position and size persistence
fn setup_window_persistence(
    _window: WindowHandle<Root>,
    _zuup_app: Arc<ZuupApp>,
) {
    // TODO: Implement window state persistence
    // For now, we'll rely on the configuration system to save/restore window state
    // This will be enhanced in future iterations when we have better GPUI event handling
    tracing::info!("Window persistence setup completed (placeholder implementation)");
}

/// Save the current window state to configuration (placeholder)
async fn _save_window_state(
    _window: WindowHandle<Root>,
    _zuup_app: Arc<ZuupApp>,
) -> anyhow::Result<()> {
    // TODO: Implement actual window state saving
    // This will be implemented when we have proper GPUI window event handling
    Ok(())
}
