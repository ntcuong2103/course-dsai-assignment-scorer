jupyter nbconvert --to script $1.ipynb
grep -v get_ipython $1.py > temp.py && mv temp.py $1.py
grep -v "# In" $1.py > temp.py && mv temp.py $1.py

