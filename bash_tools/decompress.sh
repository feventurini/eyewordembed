for f in $(find ./"$1" -name '*.gz'); do
	STEM=$(basename "${f}" .gz)
  # gunzip -c "${f}" > /decompressed_gigaword/"${STEM}"
	bzip2 -ckd "${f}" > "$2"/"${STEM}"
done