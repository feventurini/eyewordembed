src_folder="gigaword_dataset"
dest_folder="compressed_gigaword_dataset"

cd $src_folder
	for f in *; do
	tar -cvjf ../"${dest_folder}"/"${f}".tar.bz2 "${f}"
done

cd ../"${dest_folder}"
for f in *; do
	STEM="${f%%_*}"
	if ! [ -d "$STEM" ]; then
		mkdir $STEM
	fi
	mv "${f}" $STEM/
done

