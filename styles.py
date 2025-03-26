from tkinter import ttk
import tkinter as tk

class ModernTheme:
    # Colors
    PRIMARY = "#2980b9"
    SECONDARY = "#34495e"
    SUCCESS = "#2ecc71"
    DANGER = "#e74c3c"
    WARNING = "#f1c40f"
    INFO = "#3498db"
    LIGHT = "#ecf0f1"
    DARK = "#2c3e50"
    
    # Fonts
    FONT_FAMILY = "Helvetica"
    FONT_SIZES = {
        "small": 8,
        "default": 10,
        "large": 12,
        "header": 24
    }

    @staticmethod
    def apply_theme(root):
        style = ttk.Style()
        style.theme_create("modern", parent="alt", settings={
            "TLabel": {
                "configure": {
                    "background": ModernTheme.DARK,
                    "foreground": ModernTheme.LIGHT,
                    "padding": 5,
                    "font": (ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZES["default"])
                }
            },
            "TFrame": {
                "configure": {
                    "background": ModernTheme.DARK,
                    "padding": 5
                }
            },
            "TButton": {
                "configure": {
                    "background": ModernTheme.PRIMARY,
                    "foreground": ModernTheme.LIGHT,
                    "padding": (10, 5),
                    "font": (ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZES["default"]),
                },
                "map": {
                    "background": [("active", ModernTheme.INFO)],
                    "foreground": [("active", ModernTheme.LIGHT)]
                }
            },
            "TLabelframe": {
                "configure": {
                    "background": ModernTheme.DARK,
                    "foreground": ModernTheme.LIGHT,
                    "padding": 10,
                    "relief": "solid"
                }
            },
            "TCombobox": {
                "configure": {
                    "selectbackground": ModernTheme.PRIMARY,
                    "fieldbackground": ModernTheme.LIGHT,
                    "background": ModernTheme.PRIMARY,
                    "padding": 5
                }
            }
        })
        style.theme_use("modern")

class ModernWidget:
    @staticmethod
    def create_button(parent, text, command, style="primary"):
        colors = {
            "primary": (ModernTheme.PRIMARY, ModernTheme.LIGHT),
            "success": (ModernTheme.SUCCESS, ModernTheme.LIGHT),
            "danger": (ModernTheme.DANGER, ModernTheme.LIGHT),
            "warning": (ModernTheme.WARNING, ModernTheme.DARK),
            "info": (ModernTheme.INFO, ModernTheme.LIGHT)
        }
        bg_color, fg_color = colors.get(style, colors["primary"])
        
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZES["default"]),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=ModernTheme.INFO))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))
        return btn
