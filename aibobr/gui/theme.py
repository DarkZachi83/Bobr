"""Kolory, czcionki i ustawienia ekranu dla GUI."""
from __future__ import annotations

import sys

from aibobr.domain.enums import FitLevel

# Motywy (etap 13): jasny i ciemny. Słowniki COLORS / LEVEL_COLORS / VERDICT_COLORS są
# zmieniane w miejscu przez apply_style – moduły, które je zaimportowały, widzą nowe kolory
# po przebudowie okna (tak samo jak przy zmianie języka).
THEMES = ("dark", "light")
DEFAULT_THEME = "dark"
BOOTSTRAP = {"dark": "bootstrap-dark", "light": "bootstrap-light"}     # motywy ttkbootstrap 2.x

PALETTES = {
    "light": {
        "bg": "#f6f7f9", "panel": "#ffffff", "text": "#1f2328", "muted": "#5b6470", "border": "#d0d7de",
        # pasek systemu
        "vram": "#3fa66b", "ram": "#e5ad37", "missing": "#f3c2c2", "missing_line": "#c94a4a",
        # pasek modelu – kolor werdyktu, KV jaśniejszy, narzut szary
        "overhead": "#8b929b", "decoder": "#6f5bb5",          # dekoder dźwięku (mowa / muzyka)
        "marker": "#30363d", "warn": "#9a6700", "error": "#b42318",
        "track": "#eef1f4",                                   # tło pustego paska
        "on_bar": "#ffffff",                                  # napis na kolorowym odcinku
        # modele obrazów: wiersz „Naraz” (wszystko w pamięci jednocześnie) – stonowane szarości
        "soft_main": "#c9cfd6", "soft_compute": "#dde1e6", "soft_overhead": "#e4e7eb",
        "soft_encoders": "#b9c4d0", "soft_vae": "#cdd6df",
        "invalid": "#fde2e1",                                 # pole z błędem
        "select": "#cfe3ff",
    },
    "dark": {
        "bg": "#212529", "panel": "#2b3035", "text": "#e9ecef", "muted": "#a3a9b1", "border": "#45484b",
        "vram": "#3fa66b", "ram": "#d9a032", "missing": "#5c2b2b", "missing_line": "#e5534b",
        "overhead": "#7d848d", "decoder": "#8a76d6",
        "marker": "#c9d1d9", "warn": "#e3b341", "error": "#f2685f",
        "track": "#3a4046", "on_bar": "#ffffff",
        "soft_main": "#4f565f", "soft_compute": "#454a51", "soft_overhead": "#3f4349",
        "soft_encoders": "#5a626c", "soft_vae": "#4b525b",
        "invalid": "#5c2b2b", "select": "#2f5b9a",
    },
}

LEVELS = {        # (wagi, pamięć rozmowy) – na ciemnym tle jaśniejsze, żeby był kontrast
    "light": {FitLevel.GREEN: ("#1f7a45", "#7cc49a"), FitLevel.YELLOW: ("#b7791f", "#e8c27a"),
              FitLevel.RED: ("#b42318", "#ec8f88")},
    "dark": {FitLevel.GREEN: ("#2e9e5b", "#7cc49a"), FitLevel.YELLOW: ("#c98a1e", "#e8c27a"),
             FitLevel.RED: ("#d9443a", "#ec8f88")},
}
VERDICTS = {
    "light": {FitLevel.GREEN: "#1f7a45", FitLevel.YELLOW: "#9a6700", FitLevel.RED: "#b42318"},
    "dark": {FitLevel.GREEN: "#4ec27f", FitLevel.YELLOW: "#e3b341", FitLevel.RED: "#f2685f"},
}

COLORS: dict = dict(PALETTES["light"])
LEVEL_COLORS: dict = dict(LEVELS["light"])
VERDICT_COLORS: dict = dict(VERDICTS["light"])
current_theme = "light"


def theme_name(value) -> str:
    return value if value in THEMES else DEFAULT_THEME


def enable_dpi_awareness() -> None:
    """Windows ze skalowaniem 125–150%: bez tego okno tkintera jest rozmazane (SPEC §11)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def apply_style(root, theme: str = DEFAULT_THEME) -> str:
    """Ustawia motyw. Z ttkbootstrap (jeśli jest) – jego motyw; bez niego – „clam” z naszą paletą.

    Zwraca użyty motyw: "dark" albo "light"."""
    global current_theme
    from tkinter import ttk
    theme = theme_name(theme)
    COLORS.clear(); COLORS.update(PALETTES[theme])
    LEVEL_COLORS.clear(); LEVEL_COLORS.update(LEVELS[theme])
    VERDICT_COLORS.clear(); VERDICT_COLORS.update(VERDICTS[theme])
    current_theme = theme
    c = COLORS
    try:
        from ttkbootstrap.style import Style
        if Style.instance is not None and Style.instance.master is not root:
            Style.instance = None              # styl z poprzedniego (zamkniętego) okna – np. w testach
        style = Style()
        style.theme_use(BOOTSTRAP[theme])
    except Exception:                      # brak ttkbootstrap (uruchomienie z kodu) – zwykły ttk
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=c["bg"], foreground=c["text"], fieldbackground=c["panel"],
                        bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                        insertcolor=c["text"])
        style.map(".", foreground=[("disabled", c["muted"])])
        style.configure("Treeview", background=c["panel"], fieldbackground=c["panel"], foreground=c["text"])
        style.map("Treeview", background=[("selected", c["select"])])
        style.configure("TNotebook.Tab", background=c["bg"])
        style.map("TNotebook.Tab", background=[("selected", c["panel"])])
        style.map("TCombobox", fieldbackground=[("readonly", c["panel"])])
        style.configure("TButton", background=c["panel"])
    root.configure(background=c["bg"])
    # Klasyczne widżety tk (menu, lista, okna dialogowe) nie biorą stylu ttk – baza opcji:
    for pattern, value in (("*Toplevel.background", c["bg"]),
                           ("*Menu.background", c["panel"]), ("*Menu.foreground", c["text"]),
                           ("*Menu.activeBackground", c["select"]), ("*Menu.activeForeground", c["text"]),
                           ("*Menu.selectColor", c["text"]),
                           ("*Listbox.background", c["panel"]), ("*Listbox.foreground", c["text"]),
                           ("*Listbox.selectBackground", c["select"]),
                           ("*Listbox.selectForeground", c["text"])):
        root.option_add(pattern, value)
    style.configure(".", background=c["bg"], foreground=c["text"])
    # Tabele: pusta część pod wierszami (fieldbackground) – motyw jej nie ustawia, zostaje biała.
    style.configure("Treeview", background=c["panel"], fieldbackground=c["panel"], foreground=c["text"],
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
    style.configure("Panel.TFrame", background=c["panel"], relief="solid", borderwidth=1,
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
    style.configure("TNotebook.Tab", padding=(14, 5))
    style.configure("Inner.TFrame", background=c["panel"], relief="flat", borderwidth=0)
    style.configure("Panel.TLabel", background=c["panel"], foreground=c["text"])
    style.configure("Panel.TCheckbutton", background=c["panel"], foreground=c["text"])
    style.configure("Panel.TRadiobutton", background=c["panel"], foreground=c["text"])
    style.configure("Muted.TLabel", foreground=c["muted"], background=c["panel"])
    style.configure("Title.TLabel", font=("TkDefaultFont", 11, "bold"), background=c["panel"], foreground=c["text"])
    style.configure("Verdict.TLabel", font=("TkDefaultFont", 12, "bold"), background=c["panel"])
    style.configure("Note.TLabel", foreground=c["warn"], background=c["panel"])
    style.configure("Error.TLabel", foreground=c["error"], background=c["bg"])
    style.configure("Invalid.TEntry", fieldbackground=c["invalid"])
    style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))
    style.configure("Toggle.TButton", font=("TkDefaultFont", 10, "bold"), anchor="w")
    return theme
