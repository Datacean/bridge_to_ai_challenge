# Draw random figures and save as PNG
import random
from PIL import Image, ImageDraw

width, height = 800, 600
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

for _ in range(20):
	shape_type = random.choice(['rectangle', 'ellipse', 'line'])
	x1, y1 = random.randint(0, width), random.randint(0, height)
	x2, y2 = random.randint(0, width), random.randint(0, height)
	color = tuple(random.randint(0, 255) for _ in range(3))
	if shape_type == 'rectangle' or shape_type == 'ellipse':
		left, right = sorted([x1, x2])
		top, bottom = sorted([y1, y2])
		box = [left, top, right, bottom]
		if shape_type == 'rectangle':
			draw.rectangle(box, outline=color, width=3)
		else:
			draw.ellipse(box, outline=color, width=3)
	else:
		draw.line([x1, y1, x2, y2], fill=color, width=3)

image.save("random_figures.png")
