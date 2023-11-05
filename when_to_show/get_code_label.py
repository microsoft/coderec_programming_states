from tree_sitter import Language, Parser
base_path = '.'
Language.build_library(
  # Store the library in the `build` directory
  base_path + '/treesitterbuild/my-languages.so',
  # Include one or more languages
  [ 

    base_path + '/tree-sitter-python'
  ]
)

PY_LANGUAGE = Language(base_path +'/treesitterbuild/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def traverse_tree(root, code_byte):
    stack = [root]
    tokens, token_types = [], []
    
    while stack:
        node = stack.pop(0)
        if node:
            if node.type != "module":
                token = code_byte[node.start_byte:node.end_byte]
                tokens.append(token)
                token_types.append(node.type)
                # print(node.type, "****> ", token)
            if node.children:
                for child in node.children:
                    stack.append(child)
    return tokens, token_types
def parse_code(code_string):
    code_byte = tree = (bytes(code_string, "utf8"))
    tree = parser.parse(code_byte)
    cursor = tree.walk()  
    return traverse_tree(cursor.node, code_byte)


python_label_dict={
    "function def": ["function_definition", "def","class_definition ", "class"],
    "import": ["import_from_statement", "import_statement", "import"],
    "control flow": ["if_statement","elif_clause","for_statement","else_clause", "with_statement","return_statement", "with", "return","if","else","elif","while","for"],
    "error handling": ["try_statement", "except_clause","raise_statement", "try", "except","raise"],
    "test_assert": ["assert_statement"],
    "binary_operator": ["binary_operator"],
    "assignment": ["assignment"],
    "comment": ["comment"], 
    "comparison": ["comparison_operator"],
    "expression": ["expression_statement"],
    # "syntax error": ["ERROR"]
}

rev_label_dict = {}
for key in python_label_dict.keys():
    for v in python_label_dict[key]:
        rev_label_dict[v] = key
# print(rev_label_dict)  


def crude_label(prompt_token):
    # ## Create dictionary of tokens indicative of a type of prompt.

    label_dict = {"codeinit": ["!/usr/", "#!/usr/"],

                   "function def": ["def ","de f" ,"class "],
                  "test_assert": ["assert "], 
                  "import": ["import ","from "],
                  "control flow": ["if ","while ","for ","while ","else ", "elif ","return ", "with "],

                  "print":["print(", "print"], 
                  "error handling": ["try ", "catch ", "except ", "raise "],
                          "assignment":["="],
                  "comment": ["# "],
                 }

    rev_dict = {}
    for key in label_dict.keys():
        for v in label_dict[key]:
            rev_dict[v] = key
    # print(rev_dict)    


    
    if (prompt_token.strip()[0:3] == '"""'):
        return "docstring"
    # assign a label if the propmt-token contains any of the tokens in our library.
    for label in label_dict.keys():
        for token in label_dict[label]:
            if token.lower() in prompt_token.lower():
                return label 
    else:
        return "other"

def get_prompt_label(prompt_token):
    prompt_token = prompt_token.replace("de f","def")
    prompt_token = prompt_token.replace("impor t","import")
    prompt_token = prompt_token.replace("ran ge","range")
    
    if ("!/usr/" in prompt_token or "#!/usr/" in prompt_token):
        return "codeinit"
    if (prompt_token.strip()[0:3] == '"""' or prompt_token.strip()[0:3] ==  "'''" or '"""' in prompt_token or  "'''" in prompt_token):
        return "docstring"
    # parse prompt 
    tokens, token_types = parse_code(prompt_token)
    
    # assign a label if the propmt-token contains any of the tokens in our library.
    
    for label in python_label_dict.keys():
        for token_type in python_label_dict[label]:
            if token_type in token_types: 
                if token_type == "expression_statement":
                    if "print" in str(tokens[token_types.index(token_type)]):
                        return "print"
                return label
    else:
        return crude_label(prompt_token)
        # return "other"