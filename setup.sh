#FILES=
find \( -name "*.cpp" -o -name "*.h" \) -not \( -path "*/tb/*" -o) -print0 | while read -d $'\0' file
do
    echo $file
    echo $file | sed 's|^./||' | sed 's/\//_/g'
    ln -s $file $(echo $file | sed 's|^./||' | sed 's/\//_/g')
done
#| sed 's|^./||'
# for f in $FILES
# do
#   echo "Processing $f file..."
#   # take action on each file. $f store current file name
#   # cat "$f"
# done