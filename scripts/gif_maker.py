"""what the actual fuck claude"""

from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import argparse
import re


def create_stacked_gif(
    input_dir: str | Path,
    output_path: str | Path = "output.gif",
    prefix_order: list[tuple[str, str]] | None = None,
    duration: int = 500,
    loop: int = 0,
    label_width: int = 150,
    header_height: int = 50,
    font_size: int = 20,
) -> None:
    """
    Creates a GIF by stacking images at each timestep vertically.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing the images
    output_path : str | Path
        Output path for the GIF (default: "output.gif")
    prefix_order : list[tuple[str, str]] | None
        Custom order for stacking as list of (prefix, label) tuples.
        If None, sorts alphabetically by prefix and uses prefix as label.
        Example: [("model_C", "Control"), ("model_A", "Baseline"), ("model_B", "Experimental")]
    duration : int
        Duration of each frame in milliseconds (default: 500)
    loop : int
        Number of loops (0 = infinite, default: 0)
    label_width : int
        Width of the label column in pixels (default: 150)
    header_height : int
        Height of the timestep header in pixels (default: 50)
    font_size : int
        Font size for labels (default: 20)
    """
    input_dir = Path(input_dir)

    # Group images by prefix and timestep
    images_by_prefix = defaultdict(dict)
    all_timesteps = set()

    # Pattern to extract prefix and timestep from filename
    # Matches: {prefix}_epoch{timestep}.ext
    pattern = re.compile(r"(.+?)_batch000_epoch(\d+)_(?:lp|ft)\.\w+$")

    for img_path in sorted(input_dir.glob("*")):
        if img_path.is_file() and img_path.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
        ]:
            match = pattern.match(img_path.name)
            if match:
                prefix, timestep = match.groups()
                timestep = int(timestep)
                images_by_prefix[prefix][timestep] = img_path
                all_timesteps.add(timestep)

    if not images_by_prefix:
        raise ValueError(f"No images found in {input_dir} matching pattern")

    # Get all prefixes and labels in desired order
    if prefix_order is not None:
        prefixes = [p for p, _ in prefix_order]
        prefix_to_label = {p: label for p, label in prefix_order}
    else:
        prefixes = sorted(images_by_prefix.keys())
        prefix_to_label = {p: p for p in prefixes}  # Use prefix as label

    # Fill in missing timesteps by carrying forward the last valid image
    for prefix in prefixes:
        if prefix not in images_by_prefix:
            continue

        last_valid_path = None
        for t in sorted(all_timesteps):
            if t in images_by_prefix[prefix]:
                last_valid_path = images_by_prefix[prefix][t]
            elif last_valid_path is not None:
                # Carry forward the last valid image
                images_by_prefix[prefix][t] = last_valid_path

    # Load default font
    font = ImageFont.load_default(font_size)

    # Create frames by stacking images at each timestep
    frames = []
    for t in sorted(all_timesteps):
        # Collect images for this timestep
        images_at_t = []
        for prefix in prefixes:
            if prefix in images_by_prefix and t in images_by_prefix[prefix]:
                images_at_t.append((prefix, images_by_prefix[prefix][t]))

        if not images_at_t:
            continue

        # Load images
        pil_images = [Image.open(path) for _, path in images_at_t]
        prefixes_at_t = [prefix for prefix, _ in images_at_t]
        labels_at_t = [prefix_to_label[prefix] for prefix in prefixes_at_t]

        # Load images
        pil_images = [Image.open(path) for _, path in images_at_t]
        prefixes = [prefix for prefix, _ in images_at_t]

        # Get dimensions
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]

        max_width = max(widths)
        total_height = sum(heights)

        # Add space for labels if requested
        canvas_width = label_width + max_width
        canvas_height = header_height + total_height

        # Create canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
        draw = ImageDraw.Draw(canvas)

        # Add timestep header
        if header_height:
            header_text = f"Epoch {t:02d}"
            bbox = draw.textbbox((0, 0), header_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (canvas_width - text_width) // 2
            text_y = (header_height - text_height) // 2
            draw.text(
                (text_x, text_y),
                header_text,
                fill="black",
                font=ImageFont.load_default(round(1.5 * font_size)),
            )

        # Stack images with labels
        y_offset = header_height
        for img, label in zip(pil_images, labels_at_t):
            x_offset = label_width

            # Paste image
            canvas.paste(img, (x_offset, y_offset))

            # Add row label
            # Add text (centered vertically in the row)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (label_width - text_width) // 2
            text_y = y_offset + (img.height - text_height) // 2
            draw.text((text_x, text_y), label, fill="black", font=font)

            y_offset += img.height

        frames.append(canvas)

    # Save as GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
        )
        print(f"GIF created: {output_path} ({len(frames)} frames)")
    else:
        raise ValueError("No frames were created")


def main():
    parser = argparse.ArgumentParser(
        description="Create a stacked GIF from image sequences"
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input images",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output path for the generated GIF",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Duration per frame in milliseconds (default: 500)",
    )

    parser.add_argument(
        "--header_height",
        type=int,
        default=50,
        help="Height of the header in pixels (default: 50)",
    )

    parser.add_argument(
        "--font_size",
        type=int,
        default=20,
        help="Font size for labels (default: 20)",
    )

    args = parser.parse_args()

    create_stacked_gif(
        input_dir=args.input_dir,
        output_path=args.output_path,
        prefix_order=[
            ("mri", "Fixed MRI"),
            ("mri_mask", "Fixed MRI\n     Mask"),
            ("warped_hist", "  Warped\nHistology"),
            ("warped_hist_mask", "  Warped\nHistology\n     Mask"),
            ("checkerboard", "Checkerboard\n       Overlay"),
            ("canny_band", "Canny Band\n     Overlay"),
            ("canny_mask", "Canny Mask\n     Overlay"),
        ],
        duration=args.duration,
        label_width=150,
        header_height=args.header_height,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
