import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Create directory for placeholder images
os.makedirs('static/img', exist_ok=True)


# Create placeholder camera image
def create_placeholder_camera(filename, width=640, height=480):
    # Create a blank image with dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(33)  # Dark gray

    # Add text with PIL (easier text handling than OpenCV)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw camera icon
    icon_size = min(width, height) // 4
    icon_x = (width - icon_size) // 2
    icon_y = (height - icon_size) // 2 - 30

    # Draw camera body
    draw.rectangle(
        [(icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size)],
        outline=(100, 100, 100),
        width=3
    )

    # Draw camera lens
    lens_size = icon_size // 2
    lens_x = icon_x + (icon_size - lens_size) // 2
    lens_y = icon_y + (icon_size - lens_size) // 2
    draw.ellipse(
        [(lens_x, lens_y), (lens_x + lens_size, lens_y + lens_size)],
        outline=(100, 100, 100),
        width=3
    )

    # Draw text
    text = "No Camera Feed"
    text_width = draw.textlength(text, font=None)
    text_x = (width - text_width) // 2
    text_y = icon_y + icon_size + 20
    draw.text((text_x, text_y), text, fill=(180, 180, 180))

    # Convert back to OpenCV format
    img = np.array(pil_img)

    # Save the image
    cv2.imwrite(filename, img)
    print(f"Created placeholder image: {filename}")


# Create placeholder disparity map
def create_placeholder_disparity(filename, width=640, height=480):
    # Create a blank image with dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(33)  # Dark gray

    # Add text with PIL
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw disparity icon - simple gradient
    icon_size = min(width, height) // 4
    icon_x = (width - icon_size) // 2
    icon_y = (height - icon_size) // 2 - 30

    # Draw gradient box
    for i in range(icon_size):
        color_val = int(i * 255 / icon_size)
        draw.line(
            [(icon_x + i, icon_y), (icon_x + i, icon_y + icon_size)],
            fill=(color_val, color_val // 2, 255 - color_val),
            width=1
        )

    # Draw text
    text = "No Disparity Map"
    text_width = draw.textlength(text, font=None)
    text_x = (width - text_width) // 2
    text_y = icon_y + icon_size + 20
    draw.text((text_x, text_y), text, fill=(180, 180, 180))

    # Convert back to OpenCV format
    img = np.array(pil_img)

    # Save the image
    cv2.imwrite(filename, img)
    print(f"Created placeholder image: {filename}")


# Create placeholder images
create_placeholder_camera('static/img/placeholder-camera.jpg')
create_placeholder_disparity('static/img/placeholder-disparity.jpg')

print("Placeholder images created successfully.")