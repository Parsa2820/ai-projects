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
        if cell['cell_type'] == 'code':
            code += "".join(list([s for s in cell.get('source') if not re.match("\s*#.+", s)]))
            collectable_count += 1
            code+="\n\n"    
    print("your file will be graded")
    return code

if __name__ == "__main__":
   print( extract_python("../quesions.ipynb"))
