jupyter nbconvert --to script $1.ipynb
grep -v "^rm" $1.py > temp.py && mv temp.py $1.py
grep -v "^ls" $1.py > temp.py && mv temp.py $1.py
python extract_functions.py $1.py $1_fn.py

# Insert the import_lib.py at the top of the $1_fn.py
echo "from import_lib import *" | cat - $1_fn.py > temp.py && mv temp.py $1_fn.py
# grep -v get_ipython $1.py > temp.py && mv temp.py $1.py
# grep -v "^#" $1.py > temp.py && mv temp.py $1.py
# grep -v "^[a-zA-Z]" $1.py > temp.py && mv temp.py $1.py
rm -f $1.py