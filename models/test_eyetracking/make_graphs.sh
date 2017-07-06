for file in `find $1 -maxdepth 1 -not -type d`
do
	dot -Tpdf $file -o images/"$(basename "$file" ".dot")".pdf
done