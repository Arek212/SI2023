import numpy as np

plik1 = 'dane/heartdisease.txt'
plik2 = 'dane/heartdisease-type.txt'
data1 = np.loadtxt(plik1, delimiter=' ', skiprows=0, dtype=str)
data2 = np.loadtxt(plik2, delimiter=' ', skiprows=0, dtype=str)
#zad3a nie rozumiem zadania i nie wiem jak je zrobić
n, m = data1.shape
print("Zad3b")
print(n)
print("Zad3c")
for x in range(0,m -1):
      if(data2[x,1]=="n"):
            result = data1[:,x]
            min = data1[0,x]
            max = data1[0,x]
            for y in data1[:,x]:
                  if(y>max):max = y
                  if(y<min):min = y
            print("kolumna",x + 1," max: ", max, " min: ", min)
print("Zad 3d")
for x in range(0,m):
      print("kolumna",x + 1,": ", np.size(np.unique(data1[:, x])))
print("Zad 3e")
for x in range(0,m):
      print("kolumna",x + 1,": ", np.unique(data1[:, x]))
print("Zad3f")
for x in range(0,m -1):
      if(data2[x,1]=="n"):
            result = data1[:, x].astype(float)
            print("kolumna ", x,": ", np.std(result))
print("Zad4a")
data = data1
pr = 10
mis = int(n * m * pr / 100)
brak = np.random.choice(n*m, mis, replace=False)
data.ravel()[brak] = np.nan
for i in range(m):
    x = data[:, i].astype(float)
    if np.issubdtype(x.dtype, np.number):
        mean = np.nanmean(x)
        x[np.isnan(x)] = mean
    else:
        values, counts = np.unique(x, return_counts=True)
        most_common_value = values[np.argmax(counts)]
        x[np.isnan(x)] = most_common_value
print(data)

