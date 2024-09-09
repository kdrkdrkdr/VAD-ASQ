import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class AudioVisualizer:
    def __init__(self, audio_files):
        self.audio_files = audio_files
        self.current_file_index = 0
        self.load_audio()
        self.exponent = 5.5
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Speech detection parameters
        self.min_speech_duration_ms = 300
        self.min_silence_duration_ms = 100
        self.speech_pad_ms = 30
        
        self.update_plot()

    def load_audio(self):
        self.current_file = self.audio_files[self.current_file_index]
        try:
            self.audio, self.fs = sf.read(self.current_file)
            self.max_value = np.max(np.abs(self.audio))
        except Exception as e:
            print(f"Error loading file {self.current_file}: {str(e)}")
            self.audio = np.zeros(1000)  # 빈 오디오로 대체
            self.fs = 44100  # 기본 샘플링 레이트
            self.max_value = 1.0

    def detect_speech_silence(self, quantized_audio):
        is_speech = np.abs(quantized_audio) > 0
        
        min_speech_samples = int(self.min_speech_duration_ms * self.fs / 1000)
        min_silence_samples = int(self.min_silence_duration_ms * self.fs / 1000)
        pad_samples = int(self.speech_pad_ms * self.fs / 1000)
        
        speech_regions = []
        silence_regions = []
        
        state = 'silence'
        start = 0
        
        for i, speech in enumerate(is_speech):
            if state == 'silence' and speech:
                if i - start >= min_silence_samples:
                    silence_regions.append((start, i))
                start = i
                state = 'potential_speech'
            elif state == 'potential_speech' and not speech:
                if i - start >= min_speech_samples:
                    speech_regions.append((start, i))
                state = 'silence'
                start = i
        
        # 마지막 구간 처리
        if state == 'potential_speech' and len(is_speech) - start >= min_speech_samples:
            speech_regions.append((start, len(is_speech)))
        elif state == 'silence' and len(is_speech) - start >= min_silence_samples:
            silence_regions.append((start, len(is_speech)))
        
        # 사일런스 구간 병합 및 짧은 발화 제거
        merged_silence_regions = []
        for i, (silence_start, silence_end) in enumerate(silence_regions):
            if i == 0:
                merged_silence_regions.append([silence_start, silence_end])
            else:
                prev_silence_end = merged_silence_regions[-1][1]
                if silence_start - prev_silence_end < min_speech_samples:
                    # 이전 사일런스와 현재 사일런스를 연결
                    merged_silence_regions[-1][1] = silence_end
                else:
                    merged_silence_regions.append([silence_start, silence_end])
        
        # 최종 발화 구간 계산 및 pad 적용 (시작과 끝 포함)
        final_speech_regions = []
        
        # 첫 번째 발화 구간 처리 (시작 부분 pad 적용)
        if merged_silence_regions and merged_silence_regions[0][0] > 0:
            speech_end = merged_silence_regions[0][0]
            if speech_end >= min_speech_samples:
                final_speech_regions.append((
                    0,
                    min(len(is_speech), speech_end + pad_samples)
                ))
        
        # 중간 발화 구간 처리
        for i in range(len(merged_silence_regions) - 1):
            speech_start = merged_silence_regions[i][1]
            speech_end = merged_silence_regions[i+1][0]
            if speech_end - speech_start >= min_speech_samples:
                final_speech_regions.append((
                    max(0, speech_start - pad_samples),
                    min(len(is_speech), speech_end + pad_samples)
                ))
        
        # 마지막 발화 구간 처리 (끝 부분 pad 적용)
        if merged_silence_regions and merged_silence_regions[-1][1] < len(is_speech):
            speech_start = merged_silence_regions[-1][1]
            if len(is_speech) - speech_start >= min_speech_samples:
                final_speech_regions.append((
                    max(0, speech_start - pad_samples),
                    len(is_speech)
                ))
        
        print(f"발화 구간 수: {len(final_speech_regions)}")
        print(f"무음 구간 수: {len(merged_silence_regions)}")
        
        return final_speech_regions, merged_silence_regions

    def update_plot(self):
        num_levels = 2**self.exponent
        step = (2 * self.max_value) / num_levels
        self.audio_quantized = step * np.round(self.audio / step)

        t = np.arange(len(self.audio)) / self.fs
        audio_len = len(self.audio)/self.fs

        speech_regions, silence_regions = self.detect_speech_silence(self.audio_quantized)
        
        self.ax1.clear()
        self.ax2.clear()

        # Original Waveform with Speech and Silence Regions
        self.ax1.plot(t, self.audio)
        self.ax1.set_title(f'원본 파일: {os.path.basename(self.current_file)} ({self.current_file_index + 1}/{len(self.audio_files)})')
        self.ax1.set_ylabel('진폭')
        self.ax1.set_xlim(-audio_len/100, audio_len + audio_len/100)

        # 발화 구간 표시
        for start, end in speech_regions:
            self.ax1.axvspan(start/self.fs, end/self.fs, color='green', alpha=0.2)

        # Quantized Waveform
        self.ax2.plot(t, self.audio_quantized)
        self.ax2.set_title(f'양자화된 파형\n레벨: 2^{self.exponent:.1f}')
        self.ax2.set_ylabel('진폭')
        self.ax2.set_xlim(-audio_len/100, audio_len + audio_len/100)

        self.fig.tight_layout()
        self.fig.canvas.draw()

        print(f"그래프 업데이트 완료. 오디오 길이: {audio_len:.2f}초")

    def on_key(self, event):
        if event.key == 'up':
            self.exponent = min(16, self.exponent + 0.1)
            self.update_plot()
        elif event.key == 'down':
            self.exponent = max(1, self.exponent - 0.1)
            self.update_plot()
        elif event.key == 'right':
            self.current_file_index = (self.current_file_index + 1) % len(self.audio_files)
            self.load_audio()
            self.update_plot()
        elif event.key == 'left':
            self.current_file_index = (self.current_file_index - 1) % len(self.audio_files)
            self.load_audio()
            self.update_plot()

def find_audio_files(directory):
    supported_extensions = ('.wav', '.flac', '.ogg', '.aiff', '.aif', '.aifc', '.mp3')
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

if __name__ == '__main__':
    audio_dir = 'libri'
    audio_files = find_audio_files(audio_dir)
    
    if not audio_files:
        print("지원되는 오디오 파일을 찾을 수 없습니다.")
    else:
        print(f"총 {len(audio_files)}개의 오디오 파일을 찾았습니다.")
        visualizer = AudioVisualizer(audio_files)
        visualizer.fig.canvas.mpl_connect('key_press_event', visualizer.on_key)
        plt.show()
