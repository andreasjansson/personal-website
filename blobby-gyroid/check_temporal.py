import cv2
import numpy as np

def load_video_frames(video_path, max_frames=30):
    """Load frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
        count += 1
    cap.release()
    return np.array(frames)

def compute_temporal_metrics(frames):
    """Compute temporal variation metrics."""
    # Frame-to-frame differences
    diffs = np.abs(np.diff(frames, axis=0))
    mean_diff = diffs.mean()
    std_diff = diffs.std()
    max_diff = diffs.max()
    
    # Temporal standard deviation (how much each pixel varies over time)
    temporal_std = frames.std(axis=0).mean()
    
    return {
        'mean_frame_diff': mean_diff,
        'std_frame_diff': std_diff,
        'max_frame_diff': max_diff,
        'temporal_std': temporal_std,
    }

print("="*70)
print("TEMPORAL VARIATION ANALYSIS")
print("="*70)

# Analyze input video
print("\nAnalyzing INPUT video...")
input_frames = load_video_frames('input.mp4', max_frames=30)
input_metrics = compute_temporal_metrics(input_frames)
print(f"  Frames: {len(input_frames)}")
print(f"  Mean frame-to-frame diff:  {input_metrics['mean_frame_diff']:.6f}")
print(f"  Std frame-to-frame diff:   {input_metrics['std_frame_diff']:.6f}")
print(f"  Max frame-to-frame diff:   {input_metrics['max_frame_diff']:.6f}")
print(f"  Temporal std per pixel:    {input_metrics['temporal_std']:.6f}")

# Analyze output video
print("\nAnalyzing OUTPUT video (intermediate_00500.mp4)...")
try:
    output_frames = load_video_frames('intermediate_00500.mp4', max_frames=30)
    output_metrics = compute_temporal_metrics(output_frames)
    print(f"  Frames: {len(output_frames)}")
    print(f"  Mean frame-to-frame diff:  {output_metrics['mean_frame_diff']:.6f}")
    print(f"  Std frame-to-frame diff:   {output_metrics['std_frame_diff']:.6f}")
    print(f"  Max frame-to-frame diff:   {output_metrics['max_frame_diff']:.6f}")
    print(f"  Temporal std per pixel:    {output_metrics['temporal_std']:.6f}")
    
    print("\nCOMPARISON:")
    ratio_mean = output_metrics['mean_frame_diff'] / input_metrics['mean_frame_diff']
    ratio_temporal = output_metrics['temporal_std'] / input_metrics['temporal_std']
    print(f"  Frame diff ratio:    {ratio_mean:.1%} of target")
    print(f"  Temporal std ratio:  {ratio_temporal:.1%} of target")
    
    if ratio_mean < 0.3:
        print(f"  ⚠️  OUTPUT IS TOO STATIC - only {ratio_mean:.1%} movement of target!")
    if ratio_temporal < 0.3:
        print(f"  ⚠️  OUTPUT HAS NO TEMPORAL VARIATION - only {ratio_temporal:.1%} of target!")
        
except Exception as e:
    print(f"  Could not load output video: {e}")

print("\n" + "="*70)
