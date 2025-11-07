import cv2
import numpy as np
import sys

def analyze_image(img_path):
    """Analyze visual structure in an image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return None
    
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Color statistics
    mean_color = rgb.mean(axis=(0,1))
    std_color = rgb.std(axis=(0,1))
    color_range = rgb.max() - rgb.min()
    
    # Grayscale statistics
    mean_gray = gray.mean()
    std_gray = gray.std()
    
    # Edge detection (gradient magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edge_strength = edges.mean()
    strong_edges = (edges > 0.1).sum() / edges.size
    
    return {
        'mean_color': mean_color,
        'std_color': std_color,
        'color_range': color_range,
        'mean_gray': mean_gray,
        'std_gray': std_gray,
        'edge_strength': edge_strength,
        'strong_edges_pct': strong_edges * 100,
    }

# Analyze frame 0 from input video
import torch
from optimize import load_video

video, fps = load_video('input.mp4', max_frames=1)
frame0 = video[0].numpy()
cv2.imwrite('target_frame0.png', cv2.cvtColor((frame0*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

print("="*70)
print("VISUAL QUALITY ANALYSIS")
print("="*70)

target_stats = analyze_image('target_frame0.png')
print("\nTARGET (input video frame 0):")
print(f"  Mean color:      RGB({target_stats['mean_color'][0]:.3f}, {target_stats['mean_color'][1]:.3f}, {target_stats['mean_color'][2]:.3f})")
print(f"  Color std:       RGB({target_stats['std_color'][0]:.3f}, {target_stats['std_color'][1]:.3f}, {target_stats['std_color'][2]:.3f})")
print(f"  Gray std:        {target_stats['std_gray']:.4f} (higher = more contrast)")
print(f"  Edge strength:   {target_stats['edge_strength']:.4f} (higher = sharper features)")
print(f"  Strong edges:    {target_stats['strong_edges_pct']:.2f}% of pixels")

if analyze_image('frame_01000.png'):
    output_stats = analyze_image('frame_01000.png')
    print("\nOUTPUT (frame_01000.png):")
    print(f"  Mean color:      RGB({output_stats['mean_color'][0]:.3f}, {output_stats['mean_color'][1]:.3f}, {output_stats['mean_color'][2]:.3f})")
    print(f"  Color std:       RGB({output_stats['std_color'][0]:.3f}, {output_stats['std_color'][1]:.3f}, {output_stats['std_color'][2]:.3f})")
    print(f"  Gray std:        {output_stats['std_gray']:.4f}")
    print(f"  Edge strength:   {output_stats['edge_strength']:.4f}")
    print(f"  Strong edges:    {output_stats['strong_edges_pct']:.2f}% of pixels")
    
    print("\nCOMPARISON:")
    print(f"  Gray std ratio:     {output_stats['std_gray']/target_stats['std_gray']:.2%} of target")
    print(f"  Edge strength ratio: {output_stats['edge_strength']/target_stats['edge_strength']:.2%} of target")
    
    if output_stats['std_gray'] < target_stats['std_gray'] * 0.3:
        print(f"  ⚠️  OUTPUT IS TOO UNIFORM - only {output_stats['std_gray']/target_stats['std_gray']:.1%} contrast of target!")
    if output_stats['edge_strength'] < target_stats['edge_strength'] * 0.3:
        print(f"  ⚠️  OUTPUT HAS NO EDGES - only {output_stats['edge_strength']/target_stats['edge_strength']:.1%} edge strength of target!")

print("\n" + "="*70)
