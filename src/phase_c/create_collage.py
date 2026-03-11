import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import cv2
import numpy as np
import glob


def create_route_collage(input_dir="visualizations/phase_c", output_path="visualizations/phase_c/routes_collage.png"):
    """Combine all individual route images into one collage grid."""
    
    # Find all route images (exclude overview, dashboard, collage)
    route_files = sorted(glob.glob(os.path.join(input_dir, "route_*.png")))
    
    # Exclude specific routes
    exclude = ["behrakis_isec", "ruggles_station_richards"]
    route_files = [f for f in route_files if not any(ex in os.path.basename(f) for ex in exclude)]
    
    if not route_files:
        print("No route images found!")
        return
    
    print(f"Found {len(route_files)} route images")
    
    # Load all images
    images = []
    for f in route_files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
            print(f"  {os.path.basename(f)}: {img.shape}")
    
    if not images:
        return
    
    # Resize all to same dimensions
    target_h = 500
    target_w = 500
    resized = []
    for img in images:
        r = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(r)
    
    # Determine grid layout
    n = len(resized)
    if n <= 4:
        cols = 2
    elif n <= 6:
        cols = 3
    elif n <= 9:
        cols = 3
    else:
        cols = 4
    
    rows = (n + cols - 1) // cols
    
    # Pad with white images if needed
    while len(resized) < rows * cols:
        resized.append(np.ones((target_h, target_w, 3), dtype=np.uint8) * 255)
    
    # Build collage
    row_images = []
    for r in range(rows):
        row_imgs = resized[r*cols:(r+1)*cols]
        row_combined = np.hstack(row_imgs)
        row_images.append(row_combined)
    
    collage = np.vstack(row_images)
    
    # Add title bar
    title_h = 60
    title_bar = np.ones((title_h, collage.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, "NEU Campus - Building-to-Building Routes",
                (collage.shape[1]//2 - 350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    final = np.vstack([title_bar, collage])
    
    cv2.imwrite(output_path, final)
    print(f"\nCollage saved to {output_path}")
    print(f"  Size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    create_route_collage()
