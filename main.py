import sys
import argparse
from PianoSheetia import SheetConverter


def main():
    """Main entry point for piano video to MIDI conversion."""
    ap = argparse.ArgumentParser(
        description="Convert piano videos to MIDI files by analyzing key presses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('video', help='YouTube URL or local video file (.mp4)')
    ap.add_argument('-o', '--output', type=str, default='output/out.mid',
                    help='Output MIDI file name (default: output/out.mid)')
    ap.add_argument('-t', '--threshold', type=int, default=30,
                    help='Activation threshold for key press detection (default: 30)')
    ap.add_argument('--template', type=str, default='data/template/piano-88-keys-0_5.png',
                    help='Path to piano template file for detection')

    args = ap.parse_args()

    # Create converter with specified settings
    converter = SheetConverter(
        activation_threshold=args.threshold,
        template_path=args.template
    )

    # Perform the conversion
    success = converter.convert(args.video, args.output)

    if not success:
        print("Conversion failed!")
        sys.exit(1)

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
