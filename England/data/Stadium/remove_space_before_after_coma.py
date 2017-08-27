import numpy as np



file_original= open('stadiums_modified.csv','r') 
lines_original=file_original.readlines()
print(lines_original)
new_lines=[]
for line in lines_original:
    print(line)
    line=line.replace(", ", ",")
    line=line.replace(" ,", ",")
    new_lines.append(line)

f_to_write=open('stadiums_modified2.csv','w')
f.write()
    
