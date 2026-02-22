import { useState } from 'react';
import {
  Play,
  Code,
  Zap,
  AlertCircle,
  CheckCircle,
  TrendingDown,
  Loader2,
  AlertTriangle,
  Package,
  Activity,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

/* ================= TYPES ================= */

type LoopDetails = {
  total?: number;
  nested?: number;
  max_depth?: number;
};

type Bottleneck = {
  message: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  worst_case?: string;
};

type StaticAnalysis = {
  details?: {
    loops?: LoopDetails;
  };
  bottlenecks?: Bottleneck[];
  suggestions?: string[];
};

type WorstCase = {
  complexity: string;
  description: string;
  builtin_operations?: string[];
};

type AnalysisResponse = {
  predicted_complexity: string;
  confidence: number;
  worst_case?: WorstCase;
  static_analysis?: StaticAnalysis;
  required_libraries?: string[];
  // Flat fields from backend
  bottlenecks?: Bottleneck[];
  suggestions?: string[];
};

type OptimizationResponse = {
  optimized_code: string;
  explanation: string;
  improvements?: string[];
  new_complexity?: string;
  trade_offs?: string;
};

/* ================= HELPERS ================= */

const getComplexityColor = (complexity: string) => {
  const colors: Record<string, string> = {
    'O(1)':       'text-emerald-700 bg-emerald-50 border-emerald-200',
    'O(log n)':   'text-blue-700 bg-blue-50 border-blue-200',
    'O(n)':       'text-yellow-700 bg-yellow-50 border-yellow-200',
    'O(n log n)': 'text-orange-600 bg-orange-50 border-orange-200',
    'O(n^2)':     'text-red-600 bg-red-50 border-red-200',
    'O(n^3)':     'text-red-700 bg-red-100 border-red-300',
    'O(2^n)':     'text-purple-700 bg-purple-100 border-purple-300',
    'O(n!)':      'text-pink-700 bg-pink-100 border-pink-300',
  };
  return colors[complexity] ?? 'text-gray-600 bg-gray-50 border-gray-200';
};

const getWorstCaseColor = (complexity: string) => {
  const colors: Record<string, string> = {
    'O(1)':       'text-emerald-700 bg-emerald-50 border-emerald-200',
    'O(log n)':   'text-blue-700 bg-blue-50 border-blue-200',
    'O(n)':       'text-yellow-700 bg-yellow-50 border-yellow-200',
    'O(n log n)': 'text-orange-600 bg-orange-50 border-orange-200',
    'O(n²)':      'text-red-600 bg-red-50 border-red-200',
    'O(n^2)':     'text-red-600 bg-red-50 border-red-200',
    'O(n³)':      'text-red-700 bg-red-100 border-red-300',
    'O(n^3)':     'text-red-700 bg-red-100 border-red-300',
    'O(2^n)':     'text-purple-700 bg-purple-100 border-purple-300',
    'O(n!)':      'text-pink-700 bg-pink-100 border-pink-300',
  };
  return colors[complexity] ?? 'text-gray-600 bg-gray-50 border-gray-200';
};

const getSeverityIcon = (severity?: string) => {
  switch (severity) {
    case 'critical': return '🔴';
    case 'high':     return '🟠';
    case 'medium':   return '🟡';
    default:         return '🔵';
  }
};

/**
 * Render a confidence bar with a label and percentage.
 * Colour changes from red → yellow → green as confidence rises.
 */
const ConfidenceBar = ({ value }: { value: number }) => {
  const pct = Math.round(value * 100);
  const barColor =
    pct >= 80 ? 'bg-emerald-500' :
    pct >= 60 ? 'bg-yellow-400' :
    pct >= 40 ? 'bg-orange-400' : 'bg-red-400';
  const label =
    pct >= 80 ? 'High confidence' :
    pct >= 60 ? 'Moderate confidence' :
    pct >= 40 ? 'Low confidence' : 'Very low confidence';

  return (
    <div className="mt-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-medium text-gray-500">Model Confidence</span>
        <span className="text-xs font-bold text-gray-700">{pct}% — {label}</span>
      </div>
      <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden border border-gray-200">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

/* ================= COMPONENT ================= */

const ComplexityAnalyzer = () => {
  const [code, setCode]               = useState<string>('');
  const [analysis, setAnalysis]       = useState<AnalysisResponse | null>(null);
  const [optimization, setOptimization] = useState<OptimizationResponse | null>(null);
  const [loading, setLoading]         = useState<boolean>(false);
  const [optimizing, setOptimizing]   = useState<boolean>(false);
  const [error, setError]             = useState<string | null>(null);
  const [showBuiltins, setShowBuiltins] = useState<boolean>(false);

  const BACKEND_URL = 'http://localhost:5000';

  /* ================= SAMPLE PYTHON CODES ================= */

  const sampleCodes: Record<string, string> = {
    'Nested Loop': `def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates`,

    'Linear Scan': `def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val`,

    'Recursive': `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`,

    'Sorting': `import heapq\n\ndef top_k(arr, k):\n    return heapq.nlargest(k, arr)`,

    'Third-party': `import numpy as np\nimport pandas as pd\n\ndef process(df):\n    return df.sort_values('col').reset_index()`,
  };

  /* ================= API CALLS ================= */

  const analyzeCode = async () => {
    if (!code.trim()) {
      setError('Please enter some Python code to analyze');
      return;
    }
    setLoading(true);
    setError(null);
    setOptimization(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, language: 'python' })
      });
      const data: AnalysisResponse = await response.json();
      if (!response.ok) throw new Error('Analysis failed');
      setAnalysis(data);
    } catch {
      setError('Could not connect to backend. Is Flask running on port 5000?');
    } finally {
      setLoading(false);
    }
  };

  const optimizeCode = async () => {
    if (!analysis) return;
    setOptimizing(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          current_complexity: analysis.predicted_complexity,
          language: 'python'
        })
      });
      const data: OptimizationResponse = await response.json();
      if (!response.ok) throw new Error('Optimization failed');
      setOptimization(data);
    } catch {
      setError('Could not connect to optimization service.');
    } finally {
      setOptimizing(false);
    }
  };

  /* ================= DERIVED DATA ================= */

  const bottlenecks  = analysis?.bottlenecks  ?? analysis?.static_analysis?.bottlenecks  ?? [];
  const suggestions  = analysis?.suggestions  ?? analysis?.static_analysis?.suggestions  ?? [];
  const requiredLibs = analysis?.required_libraries ?? [];
  const worstCase    = analysis?.worst_case;

  /* ================= UI ================= */

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Code className="w-10 h-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800">
              Time Complexity Analyzer (Python)
            </h1>
          </div>
          <p className="text-gray-600">
            AI-powered static &amp; ML-based analysis — worst-case complexity, required libraries, and smart suggestions
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── Input Panel ── */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Python Code Input</h2>

            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Paste Python code here..."
              className="w-full h-64 p-4 border border-gray-200 rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300"
            />

            {/* Sample buttons */}
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Quick samples</p>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(sampleCodes).map(([name, snippet]) => (
                  <button
                    key={name}
                    onClick={() => { setCode(snippet); setAnalysis(null); setOptimization(null); }}
                    className="px-3 py-1 text-xs bg-gray-100 hover:bg-indigo-100 hover:text-indigo-700 rounded-md transition font-medium"
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={analyzeCode}
              disabled={loading}
              className="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white py-3 rounded-lg flex justify-center items-center gap-2 font-semibold transition"
            >
              {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <Play className="w-5 h-5" />}
              {loading ? 'Analyzing…' : 'Analyze Complexity'}
            </button>
          </div>

          {/* ── Results Panel ── */}
          <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col gap-4">
            <h2 className="text-xl font-semibold text-gray-800">Analysis Results</h2>

            {/* Error banner */}
            {error && (
              <div className="bg-red-50 border border-red-200 p-4 rounded-lg flex gap-2 text-red-700 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                {error}
              </div>
            )}

            {analysis ? (
              <>
                {/* ── Predicted Complexity ── */}
                <div className="rounded-lg border p-4 bg-gray-50">
                  <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">
                    Predicted Average-Case Complexity
                  </p>
                  <span className={`inline-block px-4 py-1.5 rounded-full font-bold text-lg border ${getComplexityColor(analysis.predicted_complexity)}`}>
                    {analysis.predicted_complexity}
                  </span>
                  <ConfidenceBar value={analysis.confidence} />
                </div>

                {/* ── Worst-Case Complexity ── */}
                {worstCase && (
                  <div className="rounded-lg border border-orange-200 bg-orange-50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-orange-500" />
                      <p className="text-xs font-semibold uppercase tracking-wide text-orange-600">
                        Worst-Case Time Complexity
                      </p>
                    </div>
                    <span className={`inline-block px-4 py-1.5 rounded-full font-bold text-lg border ${getWorstCaseColor(worstCase.complexity)}`}>
                      {worstCase.complexity}
                    </span>
                    <p className="mt-2 text-sm text-gray-600">{worstCase.description}</p>

                    {/* Built-in operation complexity notes */}
                    {worstCase.builtin_operations && worstCase.builtin_operations.length > 0 && (
                      <div className="mt-3">
                        <button
                          onClick={() => setShowBuiltins(v => !v)}
                          className="flex items-center gap-1 text-xs text-orange-700 font-medium hover:underline"
                        >
                          {showBuiltins ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          Built-in operation complexities ({worstCase.builtin_operations.length})
                        </button>
                        {showBuiltins && (
                          <ul className="mt-2 space-y-1">
                            {worstCase.builtin_operations.map((note, i) => (
                              <li key={i} className="text-xs text-gray-700 font-mono bg-white border border-orange-100 rounded px-2 py-1">
                                {note}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* ── Required Libraries ── */}
                <div className="rounded-lg border border-indigo-100 bg-indigo-50 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Package className="w-4 h-4 text-indigo-500" />
                    <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">
                      Required Third-Party Libraries
                    </p>
                  </div>
                  {requiredLibs.length > 0 ? (
                    <div className="flex flex-wrap gap-2 mt-1">
                      {requiredLibs.map((lib) => (
                        <span
                          key={lib}
                          className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800 border border-indigo-200 font-mono"
                        >
                          pip install {lib}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-indigo-700">
                      ✅ No third-party libraries — only Python stdlib used
                    </p>
                  )}
                </div>

                {/* ── Bottlenecks ── */}
                {bottlenecks.length > 0 && (
                  <div className="rounded-lg border border-red-100 bg-red-50 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-red-500" />
                      <p className="text-xs font-semibold uppercase tracking-wide text-red-600">
                        Detected Bottlenecks
                      </p>
                    </div>
                    <ul className="space-y-2">
                      {bottlenecks.map((b, i) => (
                        <li key={i} className="text-sm text-gray-700 flex gap-2">
                          <span>{getSeverityIcon(b.severity)}</span>
                          <span>
                            {b.message}
                            {b.worst_case && (
                              <span className="ml-1 text-xs text-red-500 font-semibold">
                                (worst: {b.worst_case})
                              </span>
                            )}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* ── Suggestions ── */}
                {suggestions.length > 0 && (
                  <div className="rounded-lg border border-green-100 bg-green-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-green-600 mb-2">
                      Optimization Suggestions
                    </p>
                    <ul className="space-y-1.5">
                      {suggestions.map((s, i) => (
                        <li key={i} className="text-sm text-gray-700 flex gap-2">
                          <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* ── Optimize Button ── */}
                <button
                  onClick={optimizeCode}
                  disabled={optimizing}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white py-3 rounded-lg flex justify-center items-center gap-2 font-semibold transition"
                >
                  {optimizing ? <Loader2 className="animate-spin w-5 h-5" /> : <Zap className="w-5 h-5" />}
                  {optimizing ? 'Optimizing…' : 'Generate Optimized Code'}
                </button>
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center py-16 text-gray-400">
                <Code className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-sm">Paste Python code and click <strong>Analyze Complexity</strong></p>
              </div>
            )}
          </div>
        </div>

        {/* ── Optimized Code Panel ── */}
        {optimization && (
          <div className="mt-6 bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center gap-2 mb-4">
              <TrendingDown className="text-emerald-600 w-5 h-5" />
              <h2 className="text-xl font-semibold text-gray-800">Optimized Python Code</h2>
              {optimization.new_complexity && (
                <span className={`ml-auto px-3 py-1 rounded-full text-sm font-bold border ${getComplexityColor(optimization.new_complexity)}`}>
                  {optimization.new_complexity}
                </span>
              )}
            </div>

            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto leading-relaxed">
              {optimization.optimized_code}
            </pre>

            <div className="mt-4 bg-blue-50 border border-blue-100 p-4 rounded-lg text-sm text-gray-700">
              {optimization.explanation}
            </div>

            {optimization.improvements && optimization.improvements.length > 0 && (
              <ul className="mt-4 space-y-2">
                {optimization.improvements.map((imp, i) => (
                  <li key={i} className="flex gap-2 text-sm text-gray-700">
                    <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                    {imp}
                  </li>
                ))}
              </ul>
            )}

            {optimization.trade_offs && (
              <div className="mt-4 bg-yellow-50 border border-yellow-200 p-3 rounded-lg text-sm text-yellow-800">
                <strong>Trade-offs:</strong> {optimization.trade_offs}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ComplexityAnalyzer;
