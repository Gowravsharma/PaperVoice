import os
import time
import psutil
from pdf_reader.kokoro_voice import KokoroNarrator
from pdf_reader.read_aloud import Narrator

def get_system_state():
    batt = psutil.sensors_battery()
    freq = psutil.cpu_freq()
    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    
    plugged = batt.power_plugged if batt else None
    percent = batt.percent if batt else None
    
    return {
        "plugged": plugged,
        "battery_percent": percent,
        "cpu_percent": cpu_pct,
        "cpu_freq_current": round(freq.current, 1) if freq else None,
        "cpu_freq_max": round(freq.max, 1) if freq else None,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
        "ram_total_gb": round(mem.total / (1024**3), 2),
    }

def benchmark_engine(name, narrator, sentences):
    results = []
    # Warm up 1 sentence
    narrator.chunks("Warm up sentence.")
    
    for label, text in sentences.items():
        t0 = time.perf_counter()
        chunks = narrator.chunks(text)
        dt = time.perf_counter() - t0
        
        pcm_bytes = sum(len(a) for _, a, _ in chunks)
        audio_dur = pcm_bytes / (narrator.rate * 2)
        rtf = dt / audio_dur if audio_dur > 0 else 0
        
        results.append({
            "label": label,
            "text": text,
            "words": len(text.split()),
            "compute_time": dt,
            "audio_duration": audio_dur,
            "rtf": rtf
        })
    return results

def main():
    sys_info = get_system_state()
    print("=== SYSTEM TELEMETRY ===")
    print(f"Power State: {'Plugged In (AC)' if sys_info['plugged'] else 'On Battery (DC)'}")
    print(f"Battery: {sys_info['battery_percent']}%")
    print(f"CPU Clock: {sys_info['cpu_freq_current']} MHz (Max: {sys_info['cpu_freq_max']} MHz)")
    print(f"CPU Utilization: {sys_info['cpu_percent']}% across {sys_info['cpu_cores_physical']}C/{sys_info['cpu_cores_logical']}T")
    print(f"RAM Usage: {sys_info['ram_used_gb']} GB / {sys_info['ram_total_gb']} GB\n")

    sentences = {
        "Short (6 words)": "We define the linear operator:",
        "Medium (18 words)": "Under some conditions, a stochastic representation of the solution can be established via backward stochastic differential equations.",
        "Formula (25 words)": "the absolute value of phi sub infinity, is defined as supremum over t comma x of the absolute value of phi of t comma x.",
        "Long (35 words)": "Our analysis does not require any existence and uniqueness results for the fully nonlinear partial differential equation itself, but rather relies on the regularity and stability properties of the associated forward-backward system."
    }

    print("=== BENCHMARKING KOKORO EXTREME (Default Neural Engine) ===")
    kokoro = KokoroNarrator()
    kokoro_res = benchmark_engine("Kokoro", kokoro, sentences)
    for r in kokoro_res:
        status = "Faster than playback" if r['rtf'] < 1.0 else "Slower than playback"
        print(f"[{r['label']}] ({r['words']} words)")
        print(f"  Compute Time: {r['compute_time']:.2f}s | Audio Duration: {r['audio_duration']:.2f}s | RTF: {r['rtf']:.2f} ({status})")

    print("\n=== BENCHMARKING PIPER (Classic High-Speed Engine) ===")
    piper = Narrator()
    piper_res = benchmark_engine("Piper", piper, sentences)
    for r in piper_res:
        status = "Faster than playback" if r['rtf'] < 1.0 else "Slower than playback"
        print(f"[{r['label']}] ({r['words']} words)")
        print(f"  Compute Time: {r['compute_time']:.2f}s | Audio Duration: {r['audio_duration']:.2f}s | RTF: {r['rtf']:.2f} ({status})")

if __name__ == "__main__":
    main()

