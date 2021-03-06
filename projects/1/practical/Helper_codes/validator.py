import json
from types import CellType
import re


def extract_python(addr):
    code = ""
    with open(addr, 'r') as file:
        jfile = json.load(file)
    cells = jfile['cells']
    collectable_count = 0
    code_count = 0
    for cell in cells:
        if cell['cell_type'] == 'code':
            code_count += 1
        if cell['cell_type'] == 'code' and cell['metadata'].get('collectable') == True:
            code += "".join(list([s for s in cell.get('source') if not re.match("\s*#.+", s)]))
            collectable_count += 1
            code+="\n\n"    
    if collectable_count == 26 and code_count == 43:
        print("your jupyter file will be graded")
    else:
        print("your jupyter file won't be graded")
    return code

if __name__ == "__main__":
   print( extract_python("../quesions.ipynb"))
