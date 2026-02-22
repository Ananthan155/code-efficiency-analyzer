"""
Code Parser - Extracts features from Python code using AST
FIXED VERSION FOR CP DATASET
"""

import ast
import textwrap
import re
from typing import Dict


# ================= SANITIZER =================

def sanitize_code(code):
    if not isinstance(code, str):
        return ""

    # remove tabs
    code = code.replace("\t", "    ")

    # remove windows carriage return
    code = code.replace("\r", "")

    # remove weird unicode characters from Excel
    code = re.sub(r"[^\x00-\x7F]+", "", code)

    try:
        code = textwrap.dedent(code)
    except:
        pass

    return code.strip()


# ================= PARSER =================

class CodeParser:
    """Extract 15 features from Python code for ML model"""

    def __init__(self):
        self.feature_names = [
            'num_loops',
            'max_loop_depth',
            'num_recursive_calls',
            'num_conditionals',
            'num_function_calls',
            'num_list_operations',
            'num_dict_operations',
            'num_sorting_operations',
            'num_array_accesses',
            'code_length',
            'num_nested_loops',
            'has_binary_search_pattern',
            'has_divide_conquer_pattern',
            'num_variables',
            'cyclomatic_complexity'
        ]

    def extract_features(self, code: str) -> Dict:
        """
        Extract features from CP-style Python snippets
        """

        try:

            # sanitize code
            code = sanitize_code(code)

            if not code:
                raise SyntaxError("Empty code")

            # 🚨 CRITICAL FIX FOR COMPETITIVE PROGRAMMING DATASET
            # Wrap snippet inside function so AST accepts it
            code = "def _f_():\n" + textwrap.indent(code, "    ")

            tree = ast.parse(code)

            visitor = ASTVisitor()
            visitor.visit(tree)

            features = {
                'num_loops': visitor.num_for_loops + visitor.num_while_loops,
                'max_loop_depth': visitor.max_loop_depth,
                'num_recursive_calls': visitor.num_recursive_calls,
                'num_conditionals': visitor.num_conditionals,
                'num_function_calls': visitor.num_function_calls,
                'num_list_operations': visitor.num_list_ops,
                'num_dict_operations': visitor.num_dict_ops,
                'num_sorting_operations': visitor.num_sort_ops,
                'num_array_accesses': visitor.num_array_accesses,
                'code_length': len(code.split('\n')),
                'num_nested_loops': visitor.num_nested_loops,
                'has_binary_search_pattern': int(visitor.has_binary_search),
                'has_divide_conquer_pattern': int(visitor.has_divide_conquer),
                'num_variables': len(visitor.variables),
                'cyclomatic_complexity': visitor.cyclomatic_complexity
            }

            return features

        except Exception as e:
            raise SyntaxError(str(e))

    def features_to_vector(self, features: Dict) -> list:
        return [float(features.get(name, 0)) for name in self.feature_names]


# ================= AST VISITOR =================

class ASTVisitor(ast.NodeVisitor):

    def __init__(self):
        self.num_for_loops = 0
        self.num_while_loops = 0
        self.num_conditionals = 0
        self.num_function_calls = 0
        self.num_recursive_calls = 0
        self.num_list_ops = 0
        self.num_dict_ops = 0
        self.num_sort_ops = 0
        self.num_array_accesses = 0
        self.num_nested_loops = 0
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.variables = set()
        self.has_binary_search = False
        self.has_divide_conquer = False
        self.cyclomatic_complexity = 1

    def visit_For(self, node):
        self.num_for_loops += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)

        if self.current_loop_depth > 1:
            self.num_nested_loops += 1

        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node):
        self.num_while_loops += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)

        if self.current_loop_depth > 1:
            self.num_nested_loops += 1

        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_If(self, node):
        self.num_conditionals += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.num_function_calls += 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.num_array_accesses += 1
        self.generic_visit(node)
