"""
create_logo.py - Generates MatchMyResume logo, favicon, and brand assets.
"""
import os
from PIL import Image, ImageDraw

os.makedirs("assets", exist_ok=True)

# 1. Create High-Resolution Logo (512x512)
size = 512
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Gradient Squircle Background
# Blue: #2563eb (37, 99, 235) to Cyan: #06b6d4 (6, 182, 212)
bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
bg_draw = ImageDraw.Draw(bg)

for y in range(size):
    factor = y / size
    r = int(37 * (1 - factor) + 6 * factor)
    g = int(99 * (1 - factor) + 182 * factor)
    b = int(235 * (1 - factor) + 212 * factor)
    bg_draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

# Mask for Squircle
mask = Image.new("L", (size, size), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rounded_rectangle([16, 16, size - 16, size - 16], radius=110, fill=255)

# Composite squircle
squircle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
squircle.paste(bg, (0, 0), mask)

# Inner Document Shape (White with Folded Corner)
doc_draw = ImageDraw.Draw(squircle)

# Document outline coordinates
# Main rectangle
doc_box = [130, 95, 382, 415]
doc_draw.rounded_rectangle(doc_box, radius=24, fill=(255, 255, 255, 245))

# Header avatar circle on document
doc_draw.ellipse([160, 130, 205, 175], fill=(37, 99, 235, 220))

# Header candidate name line
doc_draw.rounded_rectangle([220, 140, 350, 152], radius=6, fill=(15, 23, 42, 210))
doc_draw.rounded_rectangle([220, 160, 310, 168], radius=4, fill=(100, 116, 139, 180))

# Resume Content Lines
doc_draw.rounded_rectangle([160, 205, 350, 217], radius=6, fill=(226, 232, 240, 255))
doc_draw.rounded_rectangle([160, 235, 350, 247], radius=6, fill=(226, 232, 240, 255))
doc_draw.rounded_rectangle([160, 265, 310, 277], radius=6, fill=(226, 232, 240, 255))

# Tech Skill Pills on Document
doc_draw.rounded_rectangle([160, 305, 230, 323], radius=9, fill=(219, 234, 254, 255))  # Royal light
doc_draw.rounded_rectangle([240, 305, 320, 323], radius=9, fill=(207, 250, 254, 255))  # Cyan light
doc_draw.rounded_rectangle([160, 335, 250, 353], radius=9, fill=(209, 250, 229, 255))  # Green light

# Glowing Match Verification Badge (Bottom-Right Circle with Checkmark)
badge_center = (360, 360)
badge_radius = 65
# Shadow for badge
doc_draw.ellipse(
    [badge_center[0] - badge_radius - 4, badge_center[1] - badge_radius - 2,
     badge_center[0] + badge_radius + 4, badge_center[1] + badge_radius + 6],
    fill=(15, 23, 42, 40)
)
# Badge circle (Royal to Dark Blue)
doc_draw.ellipse(
    [badge_center[0] - badge_radius, badge_center[1] - badge_radius,
     badge_center[0] + badge_radius, badge_center[1] + badge_radius],
    fill=(16, 185, 129, 255) # Emerald green check for Match
)
# Inner white checkmark
check_points = [
    (badge_center[0] - 28, badge_center[1] - 2),
    (badge_center[0] - 8, badge_center[1] + 20),
    (badge_center[0] + 28, badge_center[1] - 22),
]
doc_draw.line(check_points, fill=(255, 255, 255, 255), width=14, joint="curve")

# Save logo.png
squircle.save("assets/logo.png", "PNG")

# 2. Save Favicon (64x64 and 32x32)
favicon = squircle.resize((64, 64), Image.Resampling.LANCZOS)
favicon.save("assets/favicon.png", "PNG")
favicon_ico = squircle.resize((32, 32), Image.Resampling.LANCZOS)
favicon_ico.save("assets/favicon.ico", "ICO")

print("Created assets/logo.png, assets/favicon.png, assets/favicon.ico successfully!")
