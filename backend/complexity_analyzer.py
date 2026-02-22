"""
Complexity Analyzer - Static code analysis
Detects patterns, bottlenecks, worst-case complexity, and generates suggestions
"""

import ast
import re

# Maps predicted complexity to worst-case description
WORST_CASE_MAP = {
    'O(1)':       {'worst': 'O(1)',       'description': 'Constant — always the same regardless of input size'},
    'O(log n)':   {'worst': 'O(n)',       'description': 'Worst case degrades to O(n) if input is unsorted or tree is unbalanced'},
    'O(n)':       {'worst': 'O(n)',       'description': 'Linear — scans entire input in worst case'},
    'O(n log n)': {'worst': 'O(n log n)', 'description': 'Quasi-linear — worst case for comparison sorts such as Merge Sort'},
    'O(n^2)':     {'worst': 'O(n^3)',     'description': 'Worst case escalates to cubic if an additional nested loop or recursion is triggered'},
    'O(n^3)':     {'worst': 'O(n^3)',     'description': 'Cubic — worst case for triple-nested iterations over input'},
    'O(2^n)':     {'worst': 'O(2^n)',     'description': 'Exponential — execution time doubles with each additional element'},
    'O(n!)':      {'worst': 'O(n!)',      'description': 'Factorial — worst case for brute-force permutation algorithms'},
}

# Known built-in / stdlib operations and their complexity notes
BUILTIN_COMPLEXITY = {
    'sort':      'O(n log n) — Timsort (worst case O(n log n))',
    'sorted':    'O(n log n) — Timsort (worst case O(n log n))',
    'min':       'O(n) — linear scan',
    'max':       'O(n) — linear scan',
    'sum':       'O(n) — linear scan',
    'set':       'O(n) to build; O(1) avg lookup',
    'dict':      'O(n) to build; O(1) avg lookup, O(n) worst-case lookup',
    'list':      'O(n) to build',
    'reversed':  'O(1) — returns iterator, no copy',
    'enumerate': 'O(1) — lazy iterator',
    'zip':       'O(1) — lazy iterator',
    'map':       'O(1) — lazy iterator',
    'filter':    'O(1) — lazy iterator',
    'any':       'O(n) worst case',
    'all':       'O(n) worst case',
    'in':        'O(n) for list/tuple; O(1) avg for set/dict',
    'bisect':    'O(log n) — binary search on sorted list',
    'heappush':  'O(log n) — heap push',
    'heappop':   'O(log n) — heap pop',
}

# Standard library modules that require no pip install
_STDLIB = {
    'os', 'sys', 'math', 'time', 'datetime', 'collections', 'itertools',
    'functools', 'typing', 're', 'json', 'random', 'pathlib', 'abc', 'io',
    'copy', 'heapq', 'bisect', 'string', 'hashlib', 'threading',
    'multiprocessing', 'subprocess', 'unittest', 'logging', 'traceback',
    'inspect', 'ast', 'pickle', 'struct', 'array', 'queue', 'socket',
    'http', 'urllib', 'email', 'html', 'xml', 'csv', 'configparser',
    'argparse', 'shutil', 'glob', 'tempfile', 'contextlib', 'dataclasses',
    'enum', 'weakref', 'gc', 'platform', 'signal', 'textwrap', 'pprint',
    'decimal', 'fractions', 'statistics', 'cmath', 'operator', 'builtins',
}


def detect_required_libraries(code: str) -> list:
    """
    Detect third-party Python libraries imported in code.
    Returns only packages that need pip install (excludes stdlib).
    """
    libs = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    libs.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    libs.add(node.module.split('.')[0])
    except Exception:
        for m in re.finditer(r'^\s*import\s+([\w]+)', code, re.MULTILINE):
            libs.add(m.group(1))
        for m in re.finditer(r'^\s*from\s+([\w]+)', code, re.MULTILINE):
            libs.add(m.group(1))

    return sorted(l for l in libs if l and l not in _STDLIB)


class ComplexityAnalyzer:
    """Static analysis for complexity detection"""

    def analyze(self, code: str) -> dict:
        """
        Analyze code and provide detailed breakdown.
        Returns: dict with predicted_complexity, worst_case, details,
                 bottlenecks, suggestions, required_libraries.
        """
        try:
            tree = ast.parse(code)
            visitor = ComplexityVisitor()
            visitor.visit(tree)

            predicted = self._derive_complexity(visitor)
            wc_info = WORST_CASE_MAP.get(predicted, {
                'worst': predicted,
                'description': 'Worst case matches predicted complexity'
            })

            builtin_notes = [
                f'{name}(): {BUILTIN_COMPLEXITY[name]}'
                for name in sorted(visitor.called_builtins)
                if name in BUILTIN_COMPLEXITY
            ]

            return {
                'predicted_complexity': predicted,
                'worst_case': {
                    'complexity': wc_info['worst'],
                    'description': wc_info['description'],
                    'builtin_operations': builtin_notes
                },
                'details': {
                    'loops': {
                        'total': visitor.num_for_loops + visitor.num_while_loops,
                        'nested': visitor.num_nested_loops,
                        'max_depth': visitor.max_loop_depth
                    },
                    'recursion': {
                        'has_recursion': visitor.num_recursive_calls > 0,
                        'recursive_calls': visitor.num_recursive_calls
                    },
                    'operations': {
                        'sorting': visitor.num_sort_ops,
                        'searching': visitor.has_binary_search,
                        'array_access': visitor.num_array_accesses
                    }
                },
                'bottlenecks': self._identify_bottlenecks(visitor),
                'suggestions': self._generate_suggestions(visitor),
                'required_libraries': detect_required_libraries(code)
            }

        except SyntaxError as e:
            return {
                'error': f'Syntax error: {str(e)}',
                'predicted_complexity': 'Unknown',
                'worst_case': {},
                'details': {},
                'bottlenecks': [],
                'suggestions': [],
                'required_libraries': detect_required_libraries(code)
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_complexity(self, v) -> str:
        """Derive predicted Big-O from visitor state"""
        if v.has_exponential_pattern:
            return 'O(2^n)'
        if v.max_loop_depth >= 3:
            return 'O(n^3)'
        if v.max_loop_depth >= 2:
            return 'O(n^2)'
        if v.num_sort_ops > 0:
            return 'O(n log n)'
        if v.has_binary_search:
            return 'O(log n)'
        if v.num_for_loops > 0 or v.num_while_loops > 0 or v.num_recursive_calls > 0:
            return 'O(n)'
        return 'O(1)'

    def _identify_bottlenecks(self, v) -> list:
        bottlenecks = []

        if v.max_loop_depth >= 3:
            bottlenecks.append({
                'type': 'triple_nested_loop',
                'severity': 'high',
                'message': 'Triple nested loops detected — O(n³) complexity',
                'worst_case': 'O(n³)'
            })
        elif v.max_loop_depth >= 2:
            bottlenecks.append({
                'type': 'nested_loop',
                'severity': 'medium',
                'message': 'Nested loops detected — O(n²) complexity',
                'worst_case': 'O(n²)'
            })

        if v.num_sort_ops > 1:
            bottlenecks.append({
                'type': 'multiple_sorts',
                'severity': 'medium',
                'message': f'{v.num_sort_ops} sort operations detected — each costs O(n log n)',
                'worst_case': 'O(n log n) per call'
            })

        if v.has_exponential_pattern:
            bottlenecks.append({
                'type': 'exponential',
                'severity': 'critical',
                'message': 'Exponential time complexity detected — O(2^n)',
                'worst_case': 'O(2^n)'
            })

        return bottlenecks

    def _generate_suggestions(self, v) -> list:
        suggestions = []

        if v.max_loop_depth >= 2:
            suggestions.append(
                'Consider using hash maps or sets to reduce nested loop complexity'
            )
        if v.num_sort_ops > 1:
            suggestions.append(
                'Multiple sorting operations can be combined or eliminated'
            )
        if v.has_exponential_pattern:
            suggestions.append(
                'Use dynamic programming or memoization to optimize recursive solutions'
            )
        if not v.has_dp_pattern and v.num_recursive_calls > 0:
            suggestions.append(
                'Consider adding memoization to avoid redundant calculations'
            )
        if v.num_list_ops > 5:
            suggestions.append(
                'Frequent list operations detected — consider more efficient data structures'
            )

        return suggestions


class ComplexityVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for complexity analysis"""

    def __init__(self):
        self.num_for_loops = 0
        self.num_while_loops = 0
        self.num_nested_loops = 0
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.num_recursive_calls = 0
        self.num_sort_ops = 0
        self.num_array_accesses = 0
        self.num_list_ops = 0
        self.has_binary_search = False
        self.has_divide_conquer = False
        self.has_dp_pattern = False
        self.has_exponential_pattern = False
        self.current_function = None
        self.called_builtins: set = set()

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
        if self._is_binary_search(node):
            self.has_binary_search = True
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_FunctionDef(self, node):
        old = self.current_function
        self.current_function = node.name
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                if 'cache' in dec.id.lower() or 'memo' in dec.id.lower():
                    self.has_dp_pattern = True
        self.generic_visit(node)
        self.current_function = old

    def visit_Call(self, node):
        # Named function calls  e.g. sorted(arr)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            self.called_builtins.add(name)

            if name == self.current_function:
                self.num_recursive_calls += 1
                # Count recursive calls within this very call node
                inner_recursive = sum(
                    1 for c in ast.walk(node)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id == self.current_function
                )
                if inner_recursive > 1:
                    self.has_exponential_pattern = True

            if name in ('sort', 'sorted'):
                self.num_sort_ops += 1
            if name in ('append', 'extend', 'insert', 'pop', 'remove'):
                self.num_list_ops += 1

        # Attribute calls  e.g. arr.sort()
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            self.called_builtins.add(attr)
            if attr in ('sort', 'sorted'):
                self.num_sort_ops += 1
            if attr in ('append', 'extend', 'insert', 'pop', 'remove'):
                self.num_list_ops += 1

        if self._is_divide_conquer(node):
            self.has_divide_conquer = True

        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.num_array_accesses += 1
        self.generic_visit(node)

    def visit_Dict(self, node):
        if self.num_for_loops > 0 or self.num_while_loops > 0:
            self.has_dp_pattern = True
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Pattern helpers
    # ------------------------------------------------------------------

    def _is_binary_search(self, node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.BinOp):
                if isinstance(child.value.op, (ast.FloorDiv, ast.Div)):
                    return True
        return False

    def _is_divide_conquer(self, node) -> bool:
        if isinstance(node.func, ast.Name):
            for arg in node.args:
                if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                    return True
        return False
