#!/usr/bin/env bash
# Batch wrapper around `party ocr` to process every XML file in a directory.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 -i INPUT_DIR -o OUTPUT_DIR -m MODEL_PATH [-d DEVICE] [-- EXTRA_ARGS...]

  -i, --input     Directory containing ALTO/PageXML files to transcribe.
  -o, --output    Destination directory for generated XML files.
  -m, --model     Path to a party safetensors model (e.g. model_stageD_11.safetensors).
  -d, --device    PyTorch device passed to party (default: cuda:0).
  --              Everything after -- is forwarded to \`party ocr\`.
  -h, --help      Show this message.
EOF
}

input_dir=""
output_dir=""
model_path=""
device="cuda:0"
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; usage; exit 1; }
            input_dir="$2"
            shift 2
            ;;
        -o|--output)
            [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; usage; exit 1; }
            output_dir="$2"
            shift 2
            ;;
        -m|--model)
            [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; usage; exit 1; }
            model_path="$2"
            shift 2
            ;;
        -d|--device)
            [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; usage; exit 1; }
            device="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$input_dir" || -z "$output_dir" || -z "$model_path" ]]; then
    echo "Error: input, output, and model paths are required." >&2
    usage
    exit 1
fi

if [[ ! -d "$input_dir" ]]; then
    echo "Error: input directory '$input_dir' does not exist." >&2
    exit 1
fi

if [[ ! -f "$model_path" ]]; then
    echo "Error: model file '$model_path' does not exist." >&2
    exit 1
fi

mkdir -p "$output_dir"

shopt -s nullglob
xml_files=("$input_dir"/*.xml)
shopt -u nullglob

if [[ ${#xml_files[@]} -eq 0 ]]; then
    echo "No XML files found in '$input_dir'." >&2
    exit 1
fi

for xml in "${xml_files[@]}"; do
    base_name=$(basename "$xml")
    if [[ "${base_name^^}" == "METS.XML" ]]; then
        echo "Skipping $base_name (ignored)."
        continue
    fi
    output_file="$output_dir/${base_name%.xml}_ocr.xml"
    echo "Processing $base_name → $(basename "$output_file")"
    party -d "$device" ocr -i "$xml" "$output_file" --load-from-file "$model_path" "${extra_args[@]}"
done
