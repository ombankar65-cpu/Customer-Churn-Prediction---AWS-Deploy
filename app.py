import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML Template with Embedded CSS styling (Shadows, layout, and responsiveness)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #1f2937;
            --text-muted: #4b5563;
            --border-color: #e5e7eb;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            padding: 40px;
            transition: transform 0.3s ease;
        }

        h2 {
            text-align: center;
            margin-bottom: 30px;
            color: var(--primary-color);
            font-weight: 700;
            font-size: 1.8rem;
            letter-spacing: -0.025em;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            body { padding: 20px 10px; }
            .container { padding: 20px; }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            margin-bottom: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        input, select {
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            font-family: inherit;
            background-color: #fafafa;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .full-width {
            grid-column: span 2;
        }
        @media (max-width: 600px) {
            .full-width { grid-column: span 1; }
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: var(--shadow-md);
            transition: all 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 12px rgba(79, 70, 229, 0.2);
        }

        button:active {
            transform: translateY(1px);
        }

        .result-container {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 1.15rem;
            display: none;
            animation: fadeIn 0.4s ease-in-out forwards;
            box-shadow: var(--shadow-md);
        }

        .churn {
            background-color: #fee2e2;
            color: #dc2626;
            border: 1px solid #fca5a5;
        }

        .loyal {
            background-color: #dcfce7;
            color: #16a34a;
            border: 1px solid #86efac;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Customer Churn Analytics Prediction</h2>
    <form id="prediction-form">
        <div class="form-grid">
            <div class="form-group">
                <label for="credit_score">Credit Score</label>
                <input type="number" id="credit_score" name="credit_score" required min="300" max="850" value="650">
            </div>

            <div class="form-group">
                <label for="country">Country</label>
                <select id="country" name="country" required>
                    <option value="0">France</option>
                    <option value="1">Germany</option>
                    <option value="2">Spain</option>
                </select>
            </div>

            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required min="18" max="100" value="35">
            </div>

            <div class="form-group">
                <label for="tenure">Tenure (Years)</label>
                <input type="number" id="tenure" name="tenure" required min="0" max="20" value="5">
            </div>

            <div class="form-group">
                <label for="balance">Account Balance ($)</label>
                <input type="number" step="0.01" id="balance" name="balance" required value="50000">
            </div>

            <div class="form-group">
                <label for="products_number">Number of Products</label>
                <input type="number" id="products_number" name="products_number" required min="1" max="4" value="1">
            </div>

            <div class="form-group">
                <label for="credit_card">Has Credit Card?</label>
                <select id="credit_card" name="credit_card" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="active_member">Is Active Member?</label>
                <select id="active_member" name="active_member" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="estimated_salary">Estimated Annual Salary ($)</label>
                <input type="number" step="0.01" id="estimated_salary" name="estimated_salary" required value="60000">
            </div>

            <div class="form-group full-width">
                <button type="submit">Analyze Churn Risk</button>
            </div>
        </div>
    </form>

    <div id="result" class="result-container"></div>
</div>

<script>
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
        });

        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            resultDiv.style.display = 'block';
            if (result.prediction === 1) {
                resultDiv.className = 'result-container churn';
                resultDiv.innerHTML = `⚠️ High Churn Risk (Probability: ${(result.probability * 100).toFixed(2)}%)`;
            } else {
                resultDiv.className = 'result-container loyal';
                resultDiv.innerHTML = `✅ Customer is Likely to Stay (Loyalty Probability: ${(result.probability * 100).toFixed(2)}%)`;
            }
        } catch (error) {
            resultDiv.style.display = 'block';
            resultDiv.className = 'result-container churn';
            resultDiv.innerHTML = 'Error processing estimation. Please check data entries.';
        }
    });
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extracted features mapping order perfectly matching model requirements
        features = [
            data['credit_score'],
            data['country'],
            data['gender'],
            data['age'],
            data['tenure'],
            data['balance'],
            data['products_number'],
            data['credit_card'],
            data['active_member'],
            data['estimated_salary']
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        probabilities = model.predict_proba(final_features)[0]
        
        # Calculate active probability score based on class output
        prob = probabilities[1] if prediction == 1 else probabilities[0]

        return jsonify({
            'prediction': int(prediction),
            'probability': float(prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
