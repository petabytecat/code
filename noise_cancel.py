import sounddevice as sd
import numpy as np

# Audio parameters
CHANNELS = 1
SAMPLE_RATE = 44100
BLOCK_SIZE = 8  # Reduced block size for lower latency
FILTER_LENGTH = 8  # Shorter filter length

class NoiseReducer:
    def __init__(self):
        self.filter_weights = np.zeros(FILTER_LENGTH)
        self.reference_buffer = np.zeros(FILTER_LENGTH)
        self.mu = 0.1  # Increased learning rate for faster adaptation

    def lms_filter(self, input_signal, reference_noise):
        # Process the signal in one go using numpy operations
        filtered = np.zeros_like(input_signal)
        error = np.zeros_like(input_signal)

        for i in range(len(input_signal)):
            # Update reference buffer
            self.reference_buffer = np.roll(self.reference_buffer, -1)
            self.reference_buffer[-1] = reference_noise[i]

            # Calculate filtered sample
            filtered[i] = np.dot(self.filter_weights, self.reference_buffer)

            # Calculate error
            error[i] = input_signal[i] - filtered[i]

            # Update weights using vectorized operation
            self.filter_weights += self.mu * error[i] * self.reference_buffer

        return error

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Get input signal
    primary = indata[:, 0]
    reference = indata[:, 1] if CHANNELS > 1 else np.random.normal(0, 0.01, len(primary))

    # Process
    cleaned = noise_reducer.lms_filter(primary, reference)

    # Output
    outdata[:] = cleaned.reshape(-1, 1)

# Initialize noise reducer
noise_reducer = NoiseReducer()

# Start audio stream with lower latency settings
try:
    with sd.Stream(channels=CHANNELS,
                  samplerate=SAMPLE_RATE,
                  blocksize=BLOCK_SIZE,
                  callback=audio_callback,
                  latency='low'):  # Request low latency
        print("Noise cancellation active. Press Ctrl+C to stop.")
        sd.sleep(2**31 - 1)
except KeyboardInterrupt:
    print("\nStopping noise cancellation.")
except Exception as e:
    print(f"Error: {e}")
