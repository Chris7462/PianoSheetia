import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic grayscale image (black background with a white rectangle)
image = cv2.imread('./data/template/piano-88-keys-1_0.png')
cv2.imshow("Image", image)
cv2.waitKey(0)      # wait for a key press
cv2.destroyAllWindows()

# Create a template (a smaller rectangle, same shape as the one in image)
template = cv2.imread('./data/videos/Reverie/out0018.png')
cv2.imshow("Template", template)
cv2.waitKey(0)      # wait for a key press
cv2.destroyAllWindows()

scaled_template = cv2.resize(template, (320, 180))
cv2.imshow("Scale Template", scaled_template)
cv2.waitKey(0)      # wait for a key press
cv2.destroyAllWindows()

# Perform template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Find best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Plot everything
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Show original image with detection
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Input Image")

# Show template
axs[1].imshow(template, cmap='gray')
axs[1].set_title("Template")

# Show result heatmap
axs[2].imshow(result, cmap='hot')
axs[2].set_title("Match Result (Heatmap)")

for ax in axs:
    ax.axis('off')

plt.show()

(min_val, max_val, min_loc, max_loc, result.shape)
