import cv2
import numpy as np

# load raw image
image = cv2.imread('./videos/Interstellar/out0018.png')  # Replace with your template path
cropped = image[280:360, :]

# Get the current dimensions of the template
h, w, _ = cropped.shape

# Calculate new dimensions (half the size)
new_width = w // 2
new_height = h // 2

# Resize the template
reduced_template = cv2.resize(cropped, (new_width, new_height))

# Save the reduced template
cv2.imwrite('reduced_template.png', reduced_template)  # Save the reduced template

# Print confirmation
print(f"Reduced template saved as 'reduced_template.jpg'.")
