import tkinter as tk
from tkinter import scrolledtext, Menu, filedialog, messagebox, simpledialog
import requests
import threading
import time
from collections import deque

class PredictiveTextEditor:
    def __init__(self, root, api_url="http://localhost:8000"):
        self.root = root
        self.root.title("LightTxt Editor with Predictive Text")
        self.root.geometry("800x600")
        
        # Apply dark theme to the root window
        self.apply_dark_theme()
        
        # API Settings
        self.api_url = api_url
        self.typing_timer = None
        self.typing_delay = 0.5  # seconds before requesting predictions
        self.current_suggestion = None
        self.suggestion_visible = False
        self.predictions_enabled = True
        
        # Latency tracking
        self.prediction_times = deque(maxlen=20)  # Track the last 20 predictions
        self.avg_latency = 0
        
        # Create UI Components
        self.create_menu()
        self.create_text_area()
        self.create_status_bar()
        
        # Initialize file state
        self.current_file = None
        self.file_modified = False
        
        # Check if prediction server is available
        self.check_server_status()
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        # Define colors
        self.bg_color = "#1E1E1E"         # Dark background
        self.text_bg_color = "#252526"     # Editor background
        self.text_color = "#D4D4D4"        # Text color
        self.accent_color = "#3C5A9A"      # Accent color
        self.highlight_bg = "#264F78"      # Selection background
        self.highlight_fg = "#FFFFFF"      # Selection text
        self.suggestion_color = "#808080"  # Suggestion text color
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
    
    def create_menu(self):
        """Create the application menu bar"""
        menu_bar = Menu(self.root, bg=self.bg_color, fg=self.text_color, activebackground=self.highlight_bg, activeforeground=self.highlight_fg)
        
        # File Menu
        file_menu = Menu(menu_bar, tearoff=0, bg=self.bg_color, fg=self.text_color, 
                         activebackground=self.highlight_bg, activeforeground=self.highlight_fg)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app, accelerator="Ctrl+Q")
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Edit Menu
        edit_menu = Menu(menu_bar, tearoff=0, bg=self.bg_color, fg=self.text_color, 
                         activebackground=self.highlight_bg, activeforeground=self.highlight_fg)
        edit_menu.add_command(label="Cut", command=self.cut_text, accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=self.copy_text, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste_text, accelerator="Ctrl+V")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        # Prediction Settings Menu
        self.pred_menu = Menu(menu_bar, tearoff=0, bg=self.bg_color, fg=self.text_color, 
                           activebackground=self.highlight_bg, activeforeground=self.highlight_fg)
        self.pred_menu.add_command(label="Disable Predictions", command=self.toggle_predictions)
        self.pred_menu.add_command(label="Set API URL", command=self.set_api_url)
        menu_bar.add_cascade(label="Predictions", menu=self.pred_menu)
        
        self.root.config(menu=menu_bar)
        
        # Key Bindings
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-q>", lambda e: self.exit_app())
    
    def create_text_area(self):
        """Create the main text editing area"""
        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            font=("Consolas", 12),
            bg=self.text_bg_color,
            fg=self.text_color,
            insertbackground=self.text_color,  # cursor color
            selectbackground=self.highlight_bg,
            selectforeground=self.highlight_fg
        )
        self.text_area.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Bind events for handling text prediction
        self.text_area.bind("<KeyRelease>", self.on_key_release)
        self.text_area.bind("<Tab>", self.accept_suggestion)
        self.text_area.bind("<Button-1>", self.clear_suggestion)  # Clear on mouse click
        
        # Track changes to detect when file is modified
        self.text_area.bind("<<Modified>>", self.on_text_modified)
        
        # Focus the text area
        self.text_area.focus_set()
    
    def create_status_bar(self):
        """Create status bar at the bottom of the window"""
        self.status_frame = tk.Frame(self.root, height=25, bg=self.bg_color)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Left side status (file info)
        self.file_status = tk.Label(
            self.status_frame, 
            text="New Document", 
            anchor=tk.W, 
            padx=5,
            bg=self.bg_color,
            fg=self.text_color
        )
        self.file_status.pack(side=tk.LEFT)
        
        # Center status (latency info)
        self.latency_status = tk.Label(
            self.status_frame, 
            text="Avg Latency: -- ms", 
            anchor=tk.CENTER, 
            padx=5,
            bg=self.bg_color,
            fg=self.text_color
        )
        self.latency_status.pack(side=tk.LEFT, expand=True)
        
        # Right side status (prediction info)
        self.prediction_status = tk.Label(
            self.status_frame, 
            text="Checking prediction server...", 
            anchor=tk.E, 
            padx=5,
            bg=self.bg_color,
            fg=self.text_color
        )
        self.prediction_status.pack(side=tk.RIGHT)
    
    def check_server_status(self):
        """Check if the prediction server is available"""
        def check():
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200 and response.json().get("status") == "up":
                    server_ready = response.json().get("model_ready", False)
                    if server_ready:
                        self.prediction_status.config(text="Predictions Ready", fg="#4EC9B0")
                    else:
                        self.prediction_status.config(text="Model Loading...", fg="#CE9178")
                        # Try again in 1 second
                        self.root.after(1000, check)
                else:
                    self.prediction_status.config(text="Prediction Server Error", fg="#F44747")
            except Exception as e:
                self.prediction_status.config(text="Prediction Server Offline", fg="#F44747")
        
        # Run in a separate thread to avoid blocking UI
        threading.Thread(target=check, daemon=True).start()
    
    def on_key_release(self, event):
        """Handle key release events for text prediction"""
        # Skip if predictions are disabled
        if not self.predictions_enabled:
            return
            
        # Skip handling special keys and only process normal typing
        if event.keysym in ('Tab', 'Return', 'Escape', 'BackSpace', 'Delete', 
                          'Left', 'Right', 'Up', 'Down', 'Home', 'End'):
            if event.keysym in ('BackSpace', 'Delete'):
                self.clear_suggestion()
            return
        
        # Clear any existing timer
        if self.typing_timer:
            self.root.after_cancel(self.typing_timer)
        
        # Set a new timer to get predictions after typing stops
        self.typing_timer = self.root.after(int(self.typing_delay * 1000), self.get_predictions)
    
    def get_predictions(self):
        """Get text predictions from the API with optimized context window"""
        # Clear any existing suggestion
        self.clear_suggestion()
        
        # Get current text cursor position
        cursor_pos = self.text_area.index(tk.INSERT)
        
        # Get the text up to the cursor
        text_up_to_cursor = self.text_area.get("1.0", cursor_pos)
        
        # Extract only the last 6 words (or fewer if not available)
        words = text_up_to_cursor.split()
        context_window = " ".join(words[-6:]) if words else ""
        
        # Ensure we have at least some text to predict from
        if not context_window.strip():
            return
        
        # Run prediction request in a separate thread to not block UI
        threading.Thread(
            target=self.request_predictions,
            args=(context_window, cursor_pos),
            daemon=True
        ).start()

    
    def request_predictions(self, text, cursor_pos):
        """Make the API request for predictions"""
        try:
            start_time = time.time()
            
            response = requests.get(
                f"{self.api_url}/predict", 
                params={"text": text, "top_k": 1},
                timeout=1
            )
            
            # Calculate request latency
            latency = time.time() - start_time
            
            if response.status_code == 200:
                # Update prediction latency stats
                self.prediction_times.append(latency * 1000)  # Convert to ms
                self.avg_latency = sum(self.prediction_times) / len(self.prediction_times)
                
                # Update the latency display in the UI thread
                self.root.after(0, lambda: self.latency_status.config(
                    text=f"Avg Latency: {self.avg_latency:.1f} ms"
                ))
                
                # Get server-side timing from response
                server_metrics = response.json().get("metadata", {})
                model_time = server_metrics.get("model_time_ms", 0)
                
                predictions = response.json().get("predictions", [])
                if predictions and len(predictions) > 0:
                    # Get the highest probability prediction
                    suggestion = predictions[0].get("word", "")
                    
                    if suggestion:
                        # Schedule displaying the suggestion in the UI thread
                        self.root.after(0, lambda: self.display_suggestion(suggestion, cursor_pos))
                        
        except Exception as e:
            print(f"Error getting predictions: {e}")
    
    def display_suggestion(self, suggestion, cursor_pos):
        """Display the suggestion inline at the cursor position"""
        # Make sure we're still at the same position
        current_pos = self.text_area.index(tk.INSERT)
        if current_pos != cursor_pos:
            return
        
        # Remember the current suggestion
        self.current_suggestion = suggestion
        
        # Insert the suggestion as grayed out text
        self.text_area.insert(tk.INSERT, suggestion)
        
        # Mark the inserted text with a tag
        end_pos = self.text_area.index(tk.INSERT)
        start_pos = f"{end_pos} - {len(suggestion)}c"
        self.text_area.tag_add("suggestion", start_pos, end_pos)
        
        # Configure the suggestion tag
        self.text_area.tag_config("suggestion", foreground=self.suggestion_color)
        
        # Move cursor back to before the suggestion
        self.text_area.mark_set(tk.INSERT, start_pos)
        
        # Remember that we have a visible suggestion
        self.suggestion_visible = True
    
    def accept_suggestion(self, event=None):
        """Accept the current suggestion when Tab is pressed"""
        if self.suggestion_visible and self.current_suggestion:
            # Remove the suggestion tag to make it regular text
            self.text_area.tag_remove("suggestion", "1.0", tk.END)
            
            # Move cursor to end of accepted suggestion
            cursor_pos = self.text_area.index(tk.INSERT)
            end_pos = f"{cursor_pos} + {len(self.current_suggestion)}c"
            self.text_area.mark_set(tk.INSERT, end_pos)
            
            # Reset suggestion state
            self.suggestion_visible = False
            self.current_suggestion = None
            
            # Force the UI to update
            self.text_area.update_idletasks()
            
            # Prevent default Tab behavior
            return "break"
    
    def clear_suggestion(self, event=None):
        """Clear any visible suggestions"""
        if self.suggestion_visible:
            # Find and remove the suggested text
            ranges = self.text_area.tag_ranges("suggestion")
            if ranges:
                self.text_area.delete(ranges[0], ranges[1])
            
            # Reset suggestion state
            self.suggestion_visible = False
            self.current_suggestion = None
    
    def toggle_predictions(self):
        """Toggle prediction functionality on/off"""
        self.predictions_enabled = not self.predictions_enabled
        if self.predictions_enabled:
            self.pred_menu.entryconfigure(0, label="Disable Predictions")
            self.prediction_status.config(text="Predictions Enabled", fg="#4EC9B0")
        else:
            self.pred_menu.entryconfigure(0, label="Enable Predictions")
            self.prediction_status.config(text="Predictions Disabled", fg="#808080")
            self.clear_suggestion()
    
    def set_api_url(self):
        """Change the API URL"""
        new_url = simpledialog.askstring(
            "API URL", 
            "Enter the URL of the prediction API:",
            initialvalue=self.api_url
        )
        if new_url:
            self.api_url = new_url
            self.check_server_status()
    
    def on_text_modified(self, event=None):
        """Handle text modified event"""
        self.file_modified = True
        self.text_area.edit_modified(False)  # Reset the modified flag
        
        # Update window title to show the modified status
        if self.current_file:
            file_name = self.current_file.split("/")[-1]
            self.root.title(f"*{file_name} - LightTxt Editor")
            self.file_status.config(text=f"*{file_name}")
        else:
            self.root.title("*Untitled - LightTxt Editor")
            self.file_status.config(text="*New Document")
    
    # File management methods
    def new_file(self):
        if self.file_modified:
            save = messagebox.askyesnocancel("Save Changes", "Save changes before creating a new file?")
            if save is None:  # Cancel
                return
            if save:  # Yes
                if not self.save_file():
                    return
        
        self.text_area.delete("1.0", tk.END)
        self.current_file = None
        self.file_modified = False
        self.root.title("Untitled - LightTxt Editor")
        self.file_status.config(text="New Document")
    
    def open_file(self):
        if self.file_modified:
            save = messagebox.askyesnocancel("Save Changes", "Save changes before opening another file?")
            if save is None:  # Cancel
                return
            if save:  # Yes
                if not self.save_file():
                    return
        
        file_path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", file_content)
            self.current_file = file_path
            self.file_modified = False
            
            file_name = file_path.split("/")[-1]
            self.root.title(f"{file_name} - LightTxt Editor")
            self.file_status.config(text=file_name)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
    
    def save_file(self):
        if not self.current_file:
            return self.save_as_file()
        
        try:
            content = self.text_area.get("1.0", tk.END)
            with open(self.current_file, "w", encoding="utf-8") as file:
                file.write(content)
            
            self.file_modified = False
            
            file_name = self.current_file.split("/")[-1]
            self.root.title(f"{file_name} - LightTxt Editor")
            self.file_status.config(text=file_name)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")
            return False
    
    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return False
        
        self.current_file = file_path
        return self.save_file()
    
    # Edit operations
    def cut_text(self):
        self.copy_text()
        self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
    
    def copy_text(self):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            # No text selected
            pass
    
    def paste_text(self):
        try:
            text = self.root.clipboard_get()
            if self.text_area.tag_ranges(tk.SEL):
                self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_area.insert(tk.INSERT, text)
        except tk.TclError:
            # No text in clipboard
            pass
    
    def exit_app(self):
        if self.file_modified:
            save = messagebox.askyesnocancel("Save Changes", "Save changes before exiting?")
            if save is None:  # Cancel
                return
            if save:  # Yes
                if not self.save_file():
                    return
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PredictiveTextEditor(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()

if __name__ == "__main__":
    main()
