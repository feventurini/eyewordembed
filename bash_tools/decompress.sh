for f in $(find ./english_gigaword/data -name '*.gz'); do
  STEM=$(basename "${f}" .gz)
  gunzip -c "${f}" > /decompressed_gigaword/"${STEM}"
done