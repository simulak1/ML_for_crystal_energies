import pandas

a=pandas.read_csv("train.csv",header=None)

print(a[2][1])
