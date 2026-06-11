from tkinter import filedialog
import os


def choose_file(parent_window=None): 
    """Opens a native system file selection dialog wrapper."""
    file_path = filedialog.askopendirectory(
        parent=parent_window,  # Pins the window cleanly on top of your app
        title="Open Lab Data File")
    return file_path if file_path != "" else None, os.basename(file_path)