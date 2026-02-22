"""
Groq Optimizer - Generate optimized Python code using Groq API
Fast and efficient code optimization with Llama models
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqOptimizer:
    """AI-powered code optimization using Groq (Llama models)"""
    
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.use_mock = not self.groq_api_key
        
        if self.use_mock:
            print("⚠️  No GROQ_API_KEY found - using mock mode for optimization")
        else:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                print("✓ Groq API client initialized")
            except Exception as e:
                print(f"⚠️  Groq initialization error: {e} - using mock mode")
                self.use_mock = True
    
    def optimize(self, code: str, current_complexity: str) -> dict:
        """
        Generate optimized version of code using Groq API
        
        Args:
            code: Original Python code
            current_complexity: Current complexity (e.g., "O(n^2)")
        
        Returns:
            dict with optimized_code, explanation, improvements, trade_offs
        """
        
        if self.use_mock:
            return self._mock_optimization(code, current_complexity)
        
        try:
            return self._optimize_with_groq(code, current_complexity)
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._mock_optimization(code, current_complexity)
    
    def _optimize_with_groq(self, code: str, current_complexity: str) -> dict:
        """Optimize using Groq API with Llama models"""
        
        try:
            prompt = self._create_prompt(code, current_complexity)
            
            # Use Groq's fastest model for code optimization
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python optimization assistant. You provide optimized code with better time complexity and clear explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Fast and powerful model
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=2000,
                top_p=1,
                stream=False
            )
            
            response_text = chat_completion.choices[0].message.content
            return self._parse_response(response_text)
        
        except Exception as e:
            print(f"Groq API call error: {e}")
            return self._mock_optimization(code, current_complexity)
    
    def _create_prompt(self, code: str, current_complexity: str) -> str:
        """Create optimization prompt for Groq"""
        
        return f"""Given this Python code with time complexity {current_complexity}, provide an optimized version with better time complexity.

Original Code:
```python
{code}
```

Current Time Complexity: {current_complexity}

Please respond in this EXACT JSON format (no markdown, just pure JSON):
{{
    "optimized_code": "the optimized Python code here",
    "explanation": "detailed explanation of optimizations",
    "new_complexity": "the improved time complexity (e.g., O(n))",
    "improvements": [
        "improvement 1",
        "improvement 2",
        "improvement 3"
    ],
    "trade_offs": "any space or other trade-offs"
}}

Focus on:
- Reducing time complexity
- Using efficient data structures (hash maps, sets, heaps)
- Eliminating redundant operations
- Applying algorithms like binary search, two pointers, sliding window
- Maintaining code readability

Respond ONLY with valid JSON, no extra text before or after."""
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse Groq API response (JSON or text)"""
        
        try:
            # Clean up response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                # Try to extract JSON from any code block
                parts = response_text.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('{') and part.endswith('}'):
                        response_text = part
                        break
            
            # Try to parse as JSON
            data = json.loads(response_text)
            
            return {
                'optimized_code': data.get('optimized_code', ''),
                'explanation': data.get('explanation', ''),
                'new_complexity': data.get('new_complexity', 'Unknown'),
                'improvements': data.get('improvements', []),
                'trade_offs': data.get('trade_offs', '')
            }
        
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Fallback: try to extract code from response
            return {
                'optimized_code': self._extract_code(response_text),
                'explanation': response_text if len(response_text) < 500 else response_text[:500] + "...",
                'new_complexity': 'Unknown',
                'improvements': ['Optimization attempted - check code'],
                'trade_offs': 'Unable to parse full response'
            }
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from text"""
        
        import re
        
        # Try to find Python code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, look for function definitions
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return text
    
    def _mock_optimization(self, code: str, current_complexity: str) -> dict:
        """Mock optimization for testing without API key"""
        
        # Pattern-based mock responses
        if 'O(n^2)' in current_complexity or 'O(n^3)' in current_complexity:
            return {
                'optimized_code': '''# Optimized using hash set approach
def optimized_solution(arr):
    """
    Optimized version using hash set for O(1) lookups
    Reduces nested loop complexity to linear time
    """
    seen = set()
    result = []
    
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result''',
                'explanation': 'Replaced nested loops with a hash set for O(1) lookups. The original code had nested iterations causing quadratic time complexity. By using a set to track seen elements, we achieve O(1) lookup time and reduce the overall complexity to O(n). This is a common optimization pattern for removing duplicates or checking membership.',
                'new_complexity': 'O(n)',
                'improvements': [
                    'Replaced nested loops with hash set data structure',
                    'Achieved O(1) lookup time instead of O(n) linear search',
                    'Single pass through the array instead of multiple passes',
                    'Eliminated redundant comparisons between elements'
                ],
                'trade_offs': 'Uses O(n) extra space for the hash set. If memory is extremely constrained, the original approach might be preferred, but in most cases the time improvement is worth the space cost.'
            }
        
        elif 'O(2^n)' in current_complexity:
            return {
                'optimized_code': '''# Optimized with dynamic programming (memoization)
def optimized_solution(n, memo=None):
    """
    Optimized version using memoization
    Caches results to avoid redundant calculations
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = optimized_solution(n-1, memo) + optimized_solution(n-2, memo)
    return memo[n]''',
                'explanation': 'Applied dynamic programming with memoization to eliminate redundant recursive calls. The original recursive approach recalculated the same values multiple times, leading to exponential time complexity. By caching previously computed results in a dictionary, each value is calculated only once, reducing time complexity from O(2^n) to O(n).',
                'new_complexity': 'O(n)',
                'improvements': [
                    'Implemented memoization dictionary to cache results',
                    'Eliminated redundant recursive calculations',
                    'Reduced exponential time to linear time',
                    'Each subproblem solved only once'
                ],
                'trade_offs': 'Uses O(n) extra space for the memoization cache. Also uses O(n) recursive call stack space. For very large n, consider iterative bottom-up DP to avoid stack overflow.'
            }
        
        elif 'O(n log n)' in current_complexity and 'sort' in code.lower():
            return {
                'optimized_code': '''# Optimized using counting sort (if range is known)
def optimized_solution(arr, max_value=None):
    """
    Optimized sorting using counting sort
    Works when range of values is known and reasonable
    """
    if not arr:
        return arr
    
    # If max_value not provided, find it
    if max_value is None:
        max_value = max(arr)
    
    # Counting sort - O(n + k) where k is range
    count = [0] * (max_value + 1)
    
    for num in arr:
        count[num] += 1
    
    result = []
    for num in range(len(count)):
        result.extend([num] * count[num])
    
    return result''',
                'explanation': 'If the range of values is known and reasonable, counting sort can achieve O(n) time complexity instead of O(n log n) comparison-based sorting. This works by counting occurrences of each value and reconstructing the sorted array. However, this only works for integers within a reasonable range.',
                'new_complexity': 'O(n + k)',
                'improvements': [
                    'Eliminated comparison-based sorting overhead',
                    'Linear time complexity when range is reasonable',
                    'Simple and efficient for integer arrays',
                    'Stable sorting algorithm'
                ],
                'trade_offs': 'Only works for integers. Requires O(k) extra space where k is the range of values. If the range is very large, this becomes impractical. Also not suitable for floating-point numbers or objects.'
            }
        
        else:
            return {
                'optimized_code': '# Code is already well-optimized\n' + code,
                'explanation': 'The current implementation already has good time complexity. The algorithm appears to be efficient for its task. Further optimization would require understanding specific use cases, input characteristics, or identifying if there are any hidden bottlenecks through profiling with real data.',
                'new_complexity': current_complexity,
                'improvements': [
                    'Code is already efficiently implemented',
                    'Current time complexity is acceptable',
                    'Consider caching if function is called repeatedly with same inputs',
                    'Profile with real data to identify actual bottlenecks'
                ],
                'trade_offs': 'No significant algorithmic improvements available without more context. Consider micro-optimizations like using built-in functions, avoiding repeated calculations, or parallel processing if dealing with large datasets.'
            }


# Optional: Function to test the optimizer
def test_groq_optimizer():
    """Test the Groq optimizer with sample code"""
    
    optimizer = GroqOptimizer()
    
    test_code = '''def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates'''
    
    print("Testing Groq Optimizer...")
    print(f"Original code:\n{test_code}\n")
    
    result = optimizer.optimize(test_code, "O(n^2)")
    
    print("="*60)
    print("OPTIMIZATION RESULT")
    print("="*60)
    print(f"\nOptimized Code:\n{result['optimized_code']}\n")
    print(f"New Complexity: {result['new_complexity']}")
    print(f"\nExplanation:\n{result['explanation']}\n")
    print("Improvements:")
    for imp in result['improvements']:
        print(f"  • {imp}")
    print(f"\nTrade-offs:\n{result['trade_offs']}")


if __name__ == '__main__':
    test_groq_optimizer()