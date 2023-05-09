input_folder="$(pwd)"
for file in "${input_folder}"/*.MOV; do
  if [ -f "$file" ]; then
    output_file="$(basename "${file}" .MOV).avi"
    ffmpeg -y -i "${file}" -c:v mjpeg -q:v 3 -huffman optimal -an "${output_file}"
  fi
done
