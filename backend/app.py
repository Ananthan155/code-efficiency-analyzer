"""
Flask Backend for Time Complexity Analyzer
Integrates with TypeScript Frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import ast as python_ast
from code_parser import CodeParser
from complexity_analyzer import ComplexityAnalyzer
from deep_forest_model import DeepForestModel
from groq_optimizer import GroqOptimizer

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize components
print("Initializing CodeParser...")
code_parser = CodeParser()
print("CodeParser OK")

print("Initializing ComplexityAnalyzer...")
complexity_analyzer = ComplexityAnalyzer()
print("ComplexityAnalyzer OK")

print("Initializing DeepForestModel...")
ml_model = DeepForestModel()
print("DeepForestModel OK")

print("Initializing GroqOptimizer...")
optimizer = GroqOptimizer()
print("GroqOptimizer OK")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask backend is running',
        'model_loaded': ml_model.is_trained
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """
    Analyze Python code for time complexity
    Request: { "code": "...", "language": "python" }
    Response: { "predicted_complexity": "O(n)", "confidence": 0.85, ... }
    """
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Validate Python syntax
        try:
            python_ast.parse(code)
        except SyntaxError as e:
            return jsonify({'error': f'Syntax error: {str(e)}'}), 400
        
        # Step 1: Extract features from code
        features = code_parser.extract_features(code)
        
        # Step 2: Predict complexity using ML model
        predicted_complexity = ml_model.predict(features)
        confidence = ml_model.get_confidence(features)
        
        # Step 3: Perform static analysis
        static_analysis = complexity_analyzer.analyze(code)
        
        response = {
            'predicted_complexity': predicted_complexity,
            'confidence': float(confidence),
            'static_analysis': static_analysis,
            'features': features
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in analyze_code: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_code():
    """
    Generate optimized version of code using Claude/OpenAI
    Request: { "code": "...", "current_complexity": "O(n^2)", "language": "python" }
    Response: { "optimized_code": "...", "explanation": "...", ... }
    """
    try:
        data = request.json
        code = data.get('code', '')
        current_complexity = data.get('current_complexity', '')
        language = data.get('language', 'python')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Generate optimized code using AI
        optimization_result = optimizer.optimize(code, current_complexity)
        
        # If optimization was generated, analyze the new code
        if optimization_result.get('optimized_code'):
            opt_code = optimization_result['optimized_code']
            try:
                opt_features = code_parser.extract_features(opt_code)
                opt_complexity = ml_model.predict(opt_features)
                optimization_result['new_complexity'] = opt_complexity
            except:
                optimization_result['new_complexity'] = 'Unable to analyze'
        
        return jsonify(optimization_result), 200
    
    except Exception as e:
        print(f"Error in optimize_code: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train/retrain the ML model with new data
    Request: { "training_data": [...] }
    """
    try:
        data = request.json
        training_data = data.get('training_data', [])
        
        if not training_data:
            return jsonify({'error': 'No training data provided'}), 400
        
        result = ml_model.train(training_data)
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error in train_model: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
