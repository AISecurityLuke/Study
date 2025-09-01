from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, render_template, request
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index() -> Any:
    chart_url = None
    if request.method == 'POST':
        categories = request.form.get('categories', '')
        values = request.form.get('values', '')
        cats = [c.strip() for c in categories.split(',') if c.strip()]
        try:
            vals = [float(v.strip()) for v in values.split(',') if v.strip()]
        except Exception:
            vals = []
        if cats and vals and len(cats) == len(vals):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(cats, vals)
            ax.set_ylabel('Value')
            ax.set_title('Bar Chart')
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            chart_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
    return render_template('index.html', chart_url=chart_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '8000')), debug=False)
