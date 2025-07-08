import ast
import sys

def extract_functions_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            start_line = node.lineno
            end_line = max(
                getattr(n, 'end_lineno', n.lineno)
                for n in ast.walk(node)
                if hasattr(n, 'lineno')
            )
            functions.append((start_line, end_line))

    return functions, source.splitlines()

def write_functions_to_file(functions, lines, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for start, end in functions:
            for line in lines[start-1:end]:
                f.write(line + '\n')
            f.write('\n')  # Add a newline between functions

if __name__ == "__main__":
    input_file = sys.argv[1]  # Input Python file
    output_file = sys.argv[2]  # Output file

    funcs, lines = extract_functions_from_file(input_file)
    write_functions_to_file(funcs, lines, output_file)
    print(f"Extracted {len(funcs)} function(s) to {output_file}")