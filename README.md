Requirements & Installation Guide

Project Description

This program features a dynamic waveform visualizer that responds to your voice in real time. Using audio input, the waveform dynamically adjusts based on the captured sound, providing a visually engaging representation of live audio signals.


External Libraries (Requires Installation)
Install all dependencies at once:

pip install numpy sounddevice torch matplotlib scipy


Setting Up a Virtual Environment (Optional but Recommended)

To keep dependencies isolated, you can set up a virtual environment before installing the libraries:

python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate  # On Windows
pip install numpy sounddevice torch matplotlib scipy

Usage

Ensure all dependencies are installed before running the program. To check installed packages:

pip list | grep "numpy\|sounddevice\|torch\|matplotlib\|scipy"

Once ready, run the program, and the waveform will respond in real time to your voice input.

Happy Coding! 🚀
