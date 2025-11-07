import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_video_frames(video_path, max_frames=50):
    """Load frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Keep RGB for analysis
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
        count += 1
    cap.release()
    return np.array(frames)

def analyze_frequency_content(frames):
    """Analyze spatial frequency content using FFT."""
    gray_frames = [cv2.cvtColor((f*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames]
    
    # Take FFT of first frame
    gray = gray_frames[0].astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Radial average to get frequency spectrum
    h, w = magnitude.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    # Bin by radius
    max_r = int(min(h, w) / 2)
    radial_profile = []
    for i in range(max_r):
        mask = (r >= i) & (r < i+1)
        if mask.any():
            radial_profile.append(magnitude[mask].mean())
    
    return np.array(radial_profile)

def analyze_motion_patterns(frames):
    """Analyze motion using optical flow."""
    gray_frames = [cv2.cvtColor((f*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames]
    
    flows = []
    for i in range(len(gray_frames)-1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    
    flows = np.array(flows)
    
    # Compute motion statistics
    magnitudes = np.sqrt(flows[:,:,:,0]**2 + flows[:,:,:,1]**2)
    
    return {
        'mean_magnitude': magnitudes.mean(),
        'max_magnitude': magnitudes.max(),
        'std_magnitude': magnitudes.std(),
        'flow_vectors': flows,
        'magnitudes': magnitudes,
    }

def analyze_spatial_structure(frame):
    """Detailed spatial structure analysis."""
    gray = cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Laplacian (second derivative - shows edges and blobs)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Local statistics (variance in local patches)
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq_mean = cv2.filter2D(gray**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    
    return {
        'gradient_mean': grad_mag.mean(),
        'gradient_std': grad_mag.std(),
        'gradient_max': grad_mag.max(),
        'laplacian_mean': np.abs(laplacian).mean(),
        'laplacian_std': laplacian.std(),
        'local_var_mean': local_var.mean(),
        'local_var_std': local_var.std(),
        'strong_edges_pct': (grad_mag > 0.1).sum() / grad_mag.size * 100,
    }

def analyze_color_distribution(frames):
    """Analyze color distribution."""
    # Per-channel statistics
    r_mean = frames[:,:,:,0].mean()
    g_mean = frames[:,:,:,1].mean()
    b_mean = frames[:,:,:,2].mean()
    
    r_std = frames[:,:,:,0].std()
    g_std = frames[:,:,:,1].std()
    b_std = frames[:,:,:,2].std()
    
    # Saturation (distance from gray)
    gray = frames.mean(axis=3, keepdims=True)
    saturation = np.abs(frames - gray).mean()
    
    return {
        'r_mean': r_mean, 'g_mean': g_mean, 'b_mean': b_mean,
        'r_std': r_std, 'g_std': g_std, 'b_std': b_std,
        'saturation': saturation,
    }

print("="*80)
print("COMPREHENSIVE VIDEO ANALYSIS")
print("="*80)

# Load videos
print("\nLoading videos...")
target_frames = load_video_frames('input.mp4', max_frames=50)
output_frames = load_video_frames('intermediate_04000.mp4', max_frames=50)

print(f"Target: {len(target_frames)} frames, shape {target_frames[0].shape}")
print(f"Output: {len(output_frames)} frames, shape {output_frames[0].shape}")

# 1. SPATIAL STRUCTURE
print("\n" + "="*80)
print("1. SPATIAL STRUCTURE (first frame)")
print("="*80)
target_spatial = analyze_spatial_structure(target_frames[0])
output_spatial = analyze_spatial_structure(output_frames[0])

print("\nTARGET:")
for key, val in target_spatial.items():
    print(f"  {key:20s}: {val:.6f}")

print("\nOUTPUT:")
for key, val in output_spatial.items():
    print(f"  {key:20s}: {val:.6f}")

print("\nRATIOS (output/target):")
for key in target_spatial:
    ratio = output_spatial[key] / (target_spatial[key] + 1e-10)
    status = "✓" if ratio > 0.5 else "⚠️"
    print(f"  {status} {key:20s}: {ratio:.1%}")

# 2. FREQUENCY CONTENT
print("\n" + "="*80)
print("2. FREQUENCY CONTENT (spatial frequencies)")
print("="*80)
target_freq = analyze_frequency_content(target_frames)
output_freq = analyze_frequency_content(output_frames)

# Compare low, mid, high frequencies
low_freq_ratio = output_freq[:5].mean() / (target_freq[:5].mean() + 1e-10)
mid_freq_ratio = output_freq[5:20].mean() / (target_freq[5:20].mean() + 1e-10)
high_freq_ratio = output_freq[20:40].mean() / (target_freq[20:40].mean() + 1e-10)

print(f"\nLow frequencies (0-5):     {low_freq_ratio:.1%} of target")
print(f"Mid frequencies (5-20):    {mid_freq_ratio:.1%} of target")
print(f"High frequencies (20-40):  {high_freq_ratio:.1%} of target")

if high_freq_ratio < 0.3:
    print("⚠️  OUTPUT IS MISSING HIGH FREQUENCIES - explains blurriness!")

# 3. MOTION ANALYSIS
print("\n" + "="*80)
print("3. MOTION ANALYSIS (optical flow)")
print("="*80)
print("Computing optical flow...")
target_motion = analyze_motion_patterns(target_frames)
output_motion = analyze_motion_patterns(output_frames)

print("\nTARGET motion:")
print(f"  Mean magnitude: {target_motion['mean_magnitude']:.6f}")
print(f"  Max magnitude:  {target_motion['max_magnitude']:.6f}")
print(f"  Std magnitude:  {target_motion['std_magnitude']:.6f}")

print("\nOUTPUT motion:")
print(f"  Mean magnitude: {output_motion['mean_magnitude']:.6f}")
print(f"  Max magnitude:  {output_motion['max_magnitude']:.6f}")
print(f"  Std magnitude:  {output_motion['std_magnitude']:.6f}")

motion_ratio = output_motion['mean_magnitude'] / (target_motion['mean_magnitude'] + 1e-10)
print(f"\nMotion magnitude ratio: {motion_ratio:.1%} of target")

if motion_ratio < 0.5:
    print("⚠️  OUTPUT HAS MUCH LESS MOTION than target!")

# Motion coherence
target_coherence = target_motion['std_magnitude'] / (target_motion['mean_magnitude'] + 1e-10)
output_coherence = output_motion['std_magnitude'] / (output_motion['mean_magnitude'] + 1e-10)
print(f"\nTarget motion coherence: {target_coherence:.3f}")
print(f"Output motion coherence: {output_coherence:.3f}")

# 4. COLOR DISTRIBUTION
print("\n" + "="*80)
print("4. COLOR DISTRIBUTION")
print("="*80)
target_color = analyze_color_distribution(target_frames)
output_color = analyze_color_distribution(output_frames)

print("\nTARGET:")
for key, val in target_color.items():
    print(f"  {key:15s}: {val:.6f}")

print("\nOUTPUT:")
for key, val in output_color.items():
    print(f"  {key:15s}: {val:.6f}")

# 5. TEMPORAL STATISTICS
print("\n" + "="*80)
print("5. TEMPORAL VARIATION PER PIXEL")
print("="*80)

target_temporal_std = target_frames.std(axis=0).mean(axis=(0,1))
output_temporal_std = output_frames.std(axis=0).mean(axis=(0,1))

print(f"\nTARGET temporal std per channel: R={target_temporal_std[0]:.6f}, G={target_temporal_std[1]:.6f}, B={target_temporal_std[2]:.6f}")
print(f"OUTPUT temporal std per channel: R={output_temporal_std[0]:.6f}, G={output_temporal_std[1]:.6f}, B={output_temporal_std[2]:.6f}")

temporal_ratio = output_temporal_std.mean() / target_temporal_std.mean()
print(f"\nTemporal variation ratio: {temporal_ratio:.1%} of target")

# SUMMARY
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✗ Spatial detail:       {output_spatial['gradient_mean']/target_spatial['gradient_mean']:.1%}")
print(f"✗ High frequencies:     {high_freq_ratio:.1%}")
print(f"✗ Motion magnitude:     {motion_ratio:.1%}")
print(f"✗ Temporal variation:   {temporal_ratio:.1%}")
print(f"✗ Strong edges:         {output_spatial['strong_edges_pct']/target_spatial['strong_edges_pct']:.1%}")

print("\n" + "="*80)
