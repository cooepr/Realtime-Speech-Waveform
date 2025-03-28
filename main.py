import numpy as np
import sounddevice as sd
import tkinter as tk
import time
import torch
import threading
from queue import Queue, Empty
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

class SpeechWaveformVAD:
    def __init__(self):
        # Audio configuration
        self.SAMPLE_RATE = 16000
        self.BLOCKSIZE = 512  # Smaller block size for more frequent updates
        self.SILENCE_THRESHOLD = 0.015  # Lower threshold to be more sensitive
        self.SILENCE_TIME = 0.2
        
        # Initialize states and buffers
        self.audio_levels = Queue()
        self.is_listening = False
        self.is_speaking = False
        self.was_speaking = False  # Track previous speaking state for transition
        self.transition_start_time = 0  # When transition from speaking to silence began
        self.in_transition = False  # Flag to indicate we're in transition period
        self.transition_period = 2.0  # Transition duration in seconds
        self.last_speech_time = time.perf_counter()
        self.transition_factor = 1.0  # Factor for scaling during transition (1.0 to 0.0)
        
        # Fixed-size waveform to make it stationary (not scrolling)
        self.display_size = 512
        
        # Current visualization data - this is what gets displayed
        self.visualization_data = np.zeros(self.display_size)
        
        # Buffer for real-time waveform display
        self.buffer_size = 100  # Number of audio blocks to display
        self.waveform_buffer = deque(maxlen=self.buffer_size)
        self.level_buffer = deque([0] * self.buffer_size, maxlen=self.buffer_size)
        
        # Persistent waveform for smooth transitions - fixed size for stationary display
        self.persistent_waveform = np.zeros(self.display_size)
        
        # For transition - store a snapshot of waveform when speech ends
        self.speech_end_waveform = np.zeros(self.display_size)
        
        # Extremely slow decay factor (99.9% retention)
        self.decay_factor = 0.999  # Extremely slow decay for gradual fadeout
        
        # Initialize with zeros
        for _ in range(self.buffer_size):
            self.waveform_buffer.append(np.zeros(self.BLOCKSIZE))
    
    def audio_callback(self, indata, frames, callback_time, status):
        """Audio input callback - captures and processes incoming audio"""
        if status:
            print(f"Audio callback status: {status}")
            return
            
        try:
            # Calculate volume normalization (loudness)
            volume_norm = np.linalg.norm(indata) * 12  # Amplified for more sensitivity
            
            # Store the audio level for visualization
            self.level_buffer.append(volume_norm)
            
            # Store raw audio for waveform visualization 
            # Reshape to ensure it fits in our fixed display size
            processed_data = indata.copy().flatten()
            
            # If the data is too large, trim it; if too small, pad with zeros
            if len(processed_data) > self.display_size:
                processed_data = processed_data[:self.display_size]
            elif len(processed_data) < self.display_size:
                padding = np.zeros(self.display_size - len(processed_data))
                processed_data = np.concatenate((processed_data, padding))
                
            self.waveform_buffer.append(processed_data)
            
            # Update VAD state
            is_speech = volume_norm > self.SILENCE_THRESHOLD
            
            # Track if we were previously speaking for state transition
            self.was_speaking = self.is_speaking
            
            if is_speech:
                self.last_speech_time = time.perf_counter()
                self.is_speaking = True
                self.in_transition = False  # Cancel any transition if speech restarts
                self.transition_factor = 1.0  # Reset transition factor
            elif time.perf_counter() - self.last_speech_time > self.SILENCE_TIME:
                # Only trigger transition on state change from speaking to not speaking
                if self.is_speaking and not self.in_transition:
                    # Start transition timer
                    self.in_transition = True
                    self.transition_start_time = time.perf_counter()
                    # Store snapshot of current waveform for smooth transition
                    self.speech_end_waveform = np.copy(self.persistent_waveform)
                    self.transition_factor = 1.0  # Start transition at full amplitude
                
                self.is_speaking = False
                
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    def start_listening(self):
        """Start the audio input stream"""
        self.is_listening = True
        
        # Start the audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=self.SAMPLE_RATE,
            channels=1,
            blocksize=self.BLOCKSIZE,
            dtype='float32'
        )
        self.stream.start()
        print("Audio stream started - listening for voice input")
    
    def stop_listening(self):
        """Stop the audio input stream"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.is_listening = False
        print("Audio stream stopped")
    
    def get_audio_level(self):
        """Get the current audio level"""
        return list(self.level_buffer)
    
    def get_waveform_data(self):
        """Get the current waveform data with smooth transitions"""
        # Get the latest audio data
        if not self.waveform_buffer:
            return np.zeros(self.display_size)
            
        current_data = self.waveform_buffer[-1]
        
        # Update persistent waveform based on state
        if self.is_speaking:
            # When speaking, more quickly adapt to the new waveform (85% new, 15% old)
            self.persistent_waveform = 0.85 * current_data + 0.15 * self.persistent_waveform
            # Set visualization directly from persistent waveform
            self.visualization_data = np.copy(self.persistent_waveform)
            
        elif self.in_transition:
            # During transition, keep the waveform shape but scale its amplitude
            elapsed = time.perf_counter() - self.transition_start_time
            
            if elapsed < self.transition_period:
                # Calculate smooth transition factor using quadratic ease-out
                progress = elapsed / self.transition_period
                ease_factor = 1 - (1 - progress) ** 2
                
                # Update the transition factor (scales from 1.0 to 0.0)
                self.transition_factor = 1.0 - ease_factor
                
                # Apply the factor to the saved speech end waveform
                self.visualization_data = self.speech_end_waveform * self.transition_factor
            else:
                # Transition finished - use the final scaled waveform
                self.visualization_data = self.speech_end_waveform * 0.0
                self.in_transition = False
                # Copy to persistent waveform for continuity with slow decay
                self.persistent_waveform = np.copy(self.visualization_data)
        else:
            # Normal silent state - apply very slow decay
            self.persistent_waveform = self.persistent_waveform * self.decay_factor
            self.visualization_data = np.copy(self.persistent_waveform)
            
        return self.visualization_data

class WaveformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Waveform Visualization")
        self.root.geometry("800x400")
        self.root.configure(bg='#000000')  # Pure black background
        
        # Create VAD analyzer
        self.vad = SpeechWaveformVAD()
        
        # IMPORTANT: Initialize all data structures before setting up visualization
        # Store the last frame's data for smoother transitions - MOVED UP
        self.last_frame_data = np.zeros(512)
        
        # Animation update interval (ms) - responsive but still smooth
        self.update_interval = 25  # Faster updates for smoother transition animation
        
        # Flag to control animation
        self.running = True
        
        # Create custom colormap for the gradient effect (white to blue)
        # Changed from purple-blue to white-blue
        colors = [(1.0, 1.0, 1.0, 0.9),    # White with high alpha
                  (0.4, 0.7, 1.0, 0.8)]    # Light blue with alpha
        
        self.cmap = LinearSegmentedColormap.from_list("voice_gradient", colors)
        
        # Previous waveform data for smooth transitions
        self.prev_waveform = np.zeros(512)
        
        # Create controls FIRST - before visualization
        self.setup_controls()
        
        # THEN set up the visualization area with animation
        self.setup_visualization()
    
    def update_plot(self, frame):
        """Update the waveform plot with new data"""
        if not self.running:
            return [self.waveform_fill, self.center_line]
            
        try:
            if self.vad.is_listening:
                # Get waveform data with transitions already applied
                waveform_data = self.vad.get_waveform_data()
                
                # Adjust smoothing based on speaking state
                if self.vad.is_speaking:
                    # Less smoothing when speaking for more details
                    smoothed_data = gaussian_filter1d(waveform_data, sigma=3)
                elif self.vad.in_transition:
                    # Keep same smoothness during transition to maintain shape
                    smoothed_data = gaussian_filter1d(waveform_data, sigma=3)
                else:
                    # More smoothing when silent for gentler appearance
                    smoothed_data = gaussian_filter1d(waveform_data, sigma=5)
                
                # Amplify for better visibility but limit height
                amplified_data = smoothed_data * 2.2
                
                # Limit maximum amplitude to keep height consistent
                max_amplitude = 0.5  # Keep at 0.5 for height consistency
                amplified_data = np.clip(amplified_data, -max_amplitude, max_amplitude)
                
                # Blend with previous frame
                if self.vad.is_speaking:
                    # 70% current data during speech for more responsiveness
                    blended_data = 0.7 * amplified_data + 0.3 * self.last_frame_data
                elif self.vad.in_transition:
                    # During transition, use less blending to make the amplitude change more apparent
                    # This is key to seeing the 2-second fade-out effect
                    blended_data = 0.9 * amplified_data + 0.1 * self.last_frame_data
                else:
                    # 55% current data when silent for smoother transitions
                    blended_data = 0.55 * amplified_data + 0.45 * self.last_frame_data
                    
                self.last_frame_data = blended_data.copy()
                
                # Update the filled area
                self.waveform_fill.remove()
                x = np.arange(len(blended_data))
                
                # Always show dynamic gradient during transition
                show_dynamic_gradient = (np.max(np.abs(blended_data)) > 0.02 or self.vad.in_transition)
                
                if show_dynamic_gradient:
                    # Calculate color values based on amplitude only (not position)
                    color_values = np.abs(blended_data) / (np.max(np.abs(blended_data)) + 0.001)
                    
                    # Create a stationary gradient based on amplitude only
                    colors = np.zeros((len(blended_data), 4))
                    for i in range(len(blended_data)):
                        # Use a fixed gradient that doesn't depend on position
                        amplitude = color_values[i]
                        
                        # Changed color blend to be primarily white with blue accents
                        # Higher amplitude = more white, lower amplitude = more blue
                        r = 0.9 + 0.1 * amplitude  # Almost white (0.9-1.0)
                        g = 0.9 + 0.1 * amplitude  # Almost white (0.9-1.0)
                        b = 1.0                    # Full blue always
                        
                        # Adjust alpha based on state
                        if self.vad.in_transition:
                            # Use transition factor to smoothly reduce alpha during transition
                            base_alpha = 0.7 * (0.3 + 0.7 * amplitude)
                            alpha = base_alpha * (0.5 + 0.5 * self.vad.transition_factor)
                        else:
                            alpha = 0.7 * (0.3 + 0.7 * amplitude)
                        
                        colors[i] = [r, g, b, alpha]
                    
                    # Create a fill_between with face colors
                    self.waveform_fill = self.ax.fill_between(
                        x, blended_data, -blended_data,
                        linewidth=0, alpha=0.9)
                    
                    # Store the faces to color them
                    collection = self.waveform_fill
                    collection.set_facecolors(colors)
                    
                    # Update the status text based on current state
                    if self.vad.is_speaking:
                        self.status_label.config(text="Speaking", fg="#00ff00")
                    elif self.vad.in_transition:
                        # Show remaining transition time (1 decimal place)
                        elapsed = time.perf_counter() - self.vad.transition_start_time
                        remaining = max(0, self.vad.transition_period - elapsed)
                        self.status_label.config(text=f"Fading... ({remaining:.1f}s)", fg="#aaaaff")
                    else:
                        self.status_label.config(text="Listening", fg="white")
                else:
                    # When nearly silent, show a minimal effect - now with light blue
                    self.waveform_fill = self.ax.fill_between(
                        x, blended_data, -blended_data,
                        color='#C0E0FF', alpha=0.4, linewidth=0)  # Light blue color
                    
                    self.status_label.config(text="Listening", fg="white")
            else:
                # When not listening, show a perfectly flat line with near-perfect retention
                # Ultra-ultra-slow decay rate (99.99% retention per frame)
                self.last_frame_data = self.last_frame_data * 0.9999  # Extremely slow decay
                
                # Update the filled area
                self.waveform_fill.remove()
                x = np.arange(len(self.last_frame_data))
                
                if np.max(np.abs(self.last_frame_data)) > 0.0001:  # Lower threshold to keep showing even longer
                    # Still showing decaying waveform
                    # Use the white-blue gradient with lower alpha
                    
                    # Calculate color values based on amplitude only (not position)
                    color_values = np.abs(self.last_frame_data) / np.max(np.abs(self.last_frame_data) + 0.001)
                    
                    # Create gradient coloring with more subtle transitions
                    colors = np.zeros((len(self.last_frame_data), 4))
                    for i in range(len(self.last_frame_data)):
                        # Fixed gradient, just amplitude based - white with blue tint
                        amplitude = color_values[i]
                        r = 0.9 + 0.1 * amplitude  # Almost white
                        g = 0.9 + 0.1 * amplitude  # Almost white
                        b = 1.0                    # Full blue
                        alpha = 0.5 * (0.3 + 0.7 * amplitude)  # Lower overall alpha for fadeout
                        colors[i] = [r, g, b, alpha]
                    
                    self.waveform_fill = self.ax.fill_between(
                        x, self.last_frame_data, -self.last_frame_data,
                        linewidth=0, alpha=0.7)
                    
                    # Store the faces to color them
                    collection = self.waveform_fill
                    collection.set_facecolors(colors)
                else:
                    # Completely flat - very light blue
                    flat_line = np.zeros(512)
                    self.waveform_fill = self.ax.fill_between(
                        x, flat_line, -flat_line, 
                        color='#E0F0FF', alpha=0.3, linewidth=0)  # Very light blue
                
                self.status_label.config(text="Ready", fg="white")
            
            return [self.waveform_fill, self.center_line]
        except Exception as e:
            print(f"Error updating plot: {e}")
            # If there's an error, ensure we still return valid artists
            return [self.waveform_fill, self.center_line]
    
    def setup_controls(self):
        """Set up control buttons"""
        self.control_frame = tk.Frame(self.root, bg='#000000')  # Pure black background
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Start button
        self.start_button = tk.Button(
            self.control_frame, text="Start Listening", 
            command=self.start_listening, bg='#303030', fg='white', 
            relief=tk.RAISED, padx=20, pady=10)
        self.start_button.pack(side=tk.LEFT, padx=20)
        
        # Stop button
        self.stop_button = tk.Button(
            self.control_frame, text="Stop Listening", 
            command=self.stop_listening, bg='#303030', fg='white', 
            relief=tk.RAISED, padx=20, pady=10, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=20)
        
        # Status label
        self.status_label = tk.Label(
            self.control_frame, text="Ready", bg='#000000', fg='white',
            font=("Arial", 12))
        self.status_label.pack(side=tk.RIGHT, padx=20)
    
    def setup_visualization(self):
        """Set up the waveform visualization"""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 2))  # Keep reduced height for compact display
        self.fig.patch.set_facecolor('#000000')  # Pure black background
        self.ax.set_facecolor('#000000')  # Pure black background
        
        # Set up axis properties
        self.ax.set_ylim(-0.6, 0.6)  # Keep at -0.6/0.6 for shorter height
        self.ax.set_xlim(0, 512)  # Match the display size
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        
        # Create a filled region instead of a line - light blue for initial state
        x = np.arange(512)
        y = np.zeros(512)
        self.waveform_fill = self.ax.fill_between(x, y, -y, color='#E0F0FF', alpha=0.3, linewidth=0)
        
        # Create a horizontal center line (very subtle)
        self.center_line, = self.ax.plot([0, 512], [0, 0], color='#333333', lw=1, alpha=0.3)
        
        # Embed the plot in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Set up animation with faster updates for smoother transition motion
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval, 
            save_count=100, blit=True, cache_frame_data=False)
    
    def start_listening(self):
        """Start listening for audio"""
        self.vad.start_listening()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Listening", fg="white")
    
    def stop_listening(self):
        """Stop listening for audio"""
        self.vad.stop_listening()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready", fg="white")
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.stop_listening()
        
        try:
            # Clean up matplotlib animation more safely
            self.ani.event_source.stop()
            self.ani._stop()
        except:
            pass
            
        try:
            plt.close(self.fig)
        except:
            pass
        
        self.root.destroy()

if __name__ == "__main__":
    # Check if CUDA is available for PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create Tkinter window
    root = tk.Tk()
    app = WaveformApp(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the Tkinter main loop
    root.mainloop()
   
