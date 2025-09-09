#![recursion_limit = "256"]

use gpui::*;
use gpui_component::{
    button::Button, Root,
    StyledExt,
};

mod config;
mod theme;
mod engine;
mod app;

use config::{ConfigManager, ThemeMode};
use theme::SystemThemeDetector;

pub struct MainView {
    current_theme: ThemeMode,
    engine_status: String,
}

impl MainView {
    pub fn new() -> Self {
        // Initialize configuration
        let mut config_manager = match ConfigManager::new() {
            Ok(manager) => manager,
            Err(e) => {
                eprintln!("Failed to create config manager: {}", e);
                ConfigManager::default()
            }
        };
        
        // Load configuration
        if let Err(e) = config_manager.load() {
            eprintln!("Failed to load configuration: {}", e);
        }
        
        let current_theme = config_manager.gui_config().theme;
        
        Self { 
            current_theme,
            engine_status: "Engine integration ready".to_string(),
        }
    }
    
    fn get_effective_theme(&self) -> ThemeMode {
        match self.current_theme {
            ThemeMode::System => SystemThemeDetector::detect_system_theme(),
            mode => mode,
        }
    }
}

impl Render for MainView {
    fn render(&mut self, _: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let effective_theme = self.get_effective_theme();
        
        div()
            .v_flex()
            .gap_4()
            .size_full()
            .items_center()
            .justify_center()
            .text_center()
            .child("Zuup Download Manager")
            .child(format!("Current theme: {:?}", self.current_theme))
            .child(format!("Effective theme: {:?}", effective_theme))
            .child(format!("Engine: {}", self.engine_status))
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .child(
                        Button::new("theme_system")
                            .label("System Theme")
                            .on_click(move |_, _, _cx| {
                                println!("System theme selected");
                            })
                    )
                    .child(
                        Button::new("theme_light")
                            .label("Light Theme")
                            .on_click(move |_, _, _cx| {
                                println!("Light theme selected");
                            })
                    )
                    .child(
                        Button::new("theme_dark")
                            .label("Dark Theme")
                            .on_click(move |_, _, _cx| {
                                println!("Dark theme selected");
                            })
                    )
            )
    }
}

fn main() {
    let app = Application::new();

    app.run(move |cx| {
        gpui_component::init(cx);

        cx.activate(true);

        let mut window_size = size(px(1200.), px(800.));
        if let Some(display) = cx.primary_display() {
            let display_size = display.bounds().size;
            window_size.width = window_size.width.min(display_size.width * 0.85);
            window_size.height = window_size.height.min(display_size.height * 0.85);
        }
        let window_bounds = Bounds::centered(None, window_size, cx);

        cx.spawn(async move |cx| {
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(window_bounds)),
                titlebar: None,
                window_min_size: Some(gpui::Size {
                    width: px(640.),
                    height: px(480.),
                }),
                kind: WindowKind::Normal,
                #[cfg(target_os = "linux")]
                window_background: gpui::WindowBackgroundAppearance::Transparent,
                #[cfg(target_os = "linux")]
                window_decorations: Some(gpui::WindowDecorations::Client),
                ..Default::default()
            };

            let window = cx
                .open_window(options, |window, cx| {
                    let view = cx.new(|_| MainView::new());
                    cx.new(|cx| Root::new(view.into(), window, cx))
                })
                .expect("failed to open window");

            window
                .update(cx, |_, window, _| {
                    window.activate_window();
                    window.set_window_title("Zuup Download Manager");
                })
                .expect("failed to update window");

            Ok::<_, anyhow::Error>(())
        })
            .detach();
    });
}
