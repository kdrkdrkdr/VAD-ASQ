import numpy as np
import soundfile as sf
import os

class AudioProcessor:
    def __init__(self, exponent=7.5):
        self.exponent = exponent  # (default=7.3) 7~11 권장
        self.max_value = 1.0

    def quantize(self, x, step):
        return step * np.round(x / step)

    def apply_fade(self, audio, fade_length, fade_type='in', steepness=2):
        x = np.linspace(-steepness, steepness, num=fade_length)
        fade = 1 / (1 + np.exp(-x))
        if fade_type == 'out':
            fade = fade[::-1]
        return audio * fade

    def process_audio(self, input_file, output_file, endpoint_padding=0.03, fade_duration=0.03, fade_steepness=2, front_s=0.5, back_s=0.45):
        audio, fs = sf.read(input_file)
        audio_len = len(audio) / fs
        num_levels = 2**self.exponent
        step = (2 * self.max_value) / num_levels
        
        audio_quantized = self.quantize(audio, step)
        non_zero = np.nonzero(np.abs(audio_quantized) > 1e-6)[0]
        if len(non_zero) > 0:
            first_change = max(0, non_zero[0] - int(endpoint_padding * fs))
            last_change = min(len(audio) - 1, non_zero[-1] + int(endpoint_padding * fs))
        else:
            first_change = 0
            last_change = len(audio) - 1

        # Fade 길이 설정
        fade_length = int(fade_duration * fs)

        # Fade in/out 적용
        fade_in_audio = self.apply_fade(audio[first_change:first_change+fade_length], fade_length, 'in', fade_steepness)
        fade_out_audio = self.apply_fade(audio[last_change-fade_length:last_change], fade_length, 'out', fade_steepness)

        # 무음 추가
        front_silence = np.zeros(int(front_s * fs))
        back_silence = np.zeros(int(back_s * fs))

        # 최종 오디오 조합
        processed_audio = np.concatenate((
            front_silence,
            fade_in_audio,
            audio[first_change+fade_length:last_change-fade_length],
            fade_out_audio,
            back_silence
        ))

        sf.write(output_file, processed_audio, fs)

def process_directory(input_dir, output_dir, exponent=7.3, use_walk=False, endpoint_padding=0.03, fade_duration=0.03, fade_steepness=2):
    processor = AudioProcessor(exponent)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if use_walk:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.wav'):
                    rel_dir = os.path.relpath(root, input_dir)
                    input_path = os.path.join(root, filename)
                    output_subdir = os.path.join(output_dir, rel_dir)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    output_path = os.path.join(output_subdir, filename)
                    processor.process_audio(input_path, output_path, endpoint_padding, fade_duration, fade_steepness)
    else:
        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                processor.process_audio(input_path, output_path, endpoint_padding, fade_duration, fade_steepness)

if __name__ == '__main__':
    input_dir = 'raw'
    output_dir = 'processed'
    process_directory(input_dir, output_dir, exponent=7.3, use_walk=True, endpoint_padding=0.03, fade_duration=0.03, fade_steepness=2)