# Build a Simple Shopify App Using Machine Learning (Python)

Most Shopify apps are CRUD dashboards.  
But adding machine learning, even in a simple way, instantly levels you up and makes your app stand out.

In this tutorial, you'll build a simple Shopify app that uses **Machine Learning with Python** to predict whether a product is likely to sell well — based on historical order data.

This is beginner-friendly ML, practical, and realistic.

---

## What You'll Build

A private Shopify app that:

- ✅ Fetches product + order data from Shopify
- ✅ Trains a simple ML model in Python
- ✅ Predicts **High / Low demand** for each product
- ✅ Returns predictions via an API endpoint

Think of it as a **"Smart Product Insights"** app.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python (FastAPI) |
| ML | scikit-learn |
| Shopify API | Admin REST API |
| Auth | Shopify Admin API token |
| Deployment | Local / Render / Railway |

---

## Step 1: Create a Shopify Custom App

1. Go to **Shopify Admin** → **Settings** → **Apps and sales channels**
2. Click **Develop apps** → **Create an app**
3. Name it `Smart Product Insights`
4. Enable **Admin API access**
5. Give access to:
   - ✅ **Products** (read)
   - ✅ **Orders** (read)
6. **Save your Admin API access token.**

---

## Step 2: Set Up the Python Backend

Create a project folder:

```bash
mkdir shopify-ml-app
cd shopify-ml-app
```

Create a virtual environment:

```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn requests pandas scikit-learn python-dotenv
```

Or create a `requirements.txt`:

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
requests>=2.31.0
pandas>=2.2.0
scikit-learn>=1.4.0
python-dotenv>=1.0.0
```

Then run:

```bash
pip install -r requirements.txt
```

---

## Step 3: Create the Configuration

Create a `.env` file for your Shopify credentials:

```env
SHOPIFY_STORE_URL=your-store-name.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ Never commit your `.env` file to version control!

---

## Step 4: Create the FastAPI App

Create `main.py`:

```python
"""
Smart Product Insights - Shopify ML App
========================================
A simple Shopify app that uses Machine Learning to predict 
whether a product is likely to sell well based on historical order data.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Product Insights",
    description="ML-powered product demand prediction for Shopify stores",
    version="1.0.0"
)

# Shopify Configuration
SHOP = os.getenv("SHOPIFY_STORE_URL", "your-store-name.myshopify.com")
TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
HEADERS = {
    "X-Shopify-Access-Token": TOKEN,
    "Content-Type": "application/json"
}

# Shopify API version
API_VERSION = "2024-01"
```

---

## Step 5: Fetch Shopify Order Data

Add a helper function to fetch orders:

```python
def fetch_orders(limit: int = 250) -> pd.DataFrame:
    """
    Fetch order data from Shopify Admin API.
    
    Args:
        limit: Maximum number of orders to fetch (max 250 per request)
    
    Returns:
        DataFrame with product_id, quantity, and price for each line item
    """
    url = f"https://{SHOP}/admin/api/{API_VERSION}/orders.json?status=any&limit={limit}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch orders from Shopify: {str(e)}")
    
    orders = response.json().get("orders", [])
    
    if not orders:
        raise HTTPException(status_code=404, detail="No orders found in your Shopify store")
    
    rows = []
    for order in orders:
        for item in order.get("line_items", []):
            # Skip items without product_id (deleted products, custom items, etc.)
            if item.get("product_id"):
                rows.append({
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "price": float(item["price"])
                })
    
    if not rows:
        raise HTTPException(status_code=404, detail="No product line items found in orders")
    
    return pd.DataFrame(rows)
```

Add a function to fetch product names:

```python
def fetch_products() -> dict:
    """
    Fetch product data from Shopify to get product names.
    
    Returns:
        Dictionary mapping product_id to product title
    """
    url = f"https://{SHOP}/admin/api/{API_VERSION}/products.json?limit=250"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {}  # Return empty dict if we can't fetch products
    
    products = response.json().get("products", [])
    return {p["id"]: p["title"] for p in products}
```

---

## Step 6: Prepare Data for ML

We'll label products as:
- `1` → **High demand** (above median quantity sold)
- `0` → **Low demand** (below median quantity sold)

```python
def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare and label data for ML training.
    
    Products are labeled as:
        1 → High demand (above median quantity sold)
        0 → Low demand (below median quantity sold)
    
    Args:
        df: DataFrame with order line items
    
    Returns:
        Tuple of (features, labels, grouped_data)
    """
    # Group by product and aggregate
    grouped = df.groupby("product_id").agg({
        "quantity": "sum",      # Total units sold
        "price": "mean"         # Average price
    }).reset_index()
    
    # Label products based on median quantity
    median_quantity = grouped["quantity"].median()
    grouped["high_demand"] = (grouped["quantity"] > median_quantity).astype(int)
    
    # Prepare features and labels
    X = grouped[["quantity", "price"]]
    y = grouped["high_demand"]
    
    return X, y, grouped
```

---

## Step 7: Train a Simple ML Model

```python
def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression model for demand prediction.
    
    Args:
        X: Feature matrix (quantity, price)
        y: Labels (0 or 1)
    
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model
```

This model predicts whether a product is likely high demand or not.

---

## Step 8: Create API Endpoints

### Root Endpoint

```python
@app.get("/")
def root():
    """Health check and welcome endpoint."""
    return {
        "app": "Smart Product Insights",
        "status": "running",
        "endpoints": {
            "/predict": "Get demand predictions for all products",
            "/stats": "Get order statistics",
            "/health": "Health check"
        }
    }
```

### Health Check

```python
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "shop": SHOP}
```

### Main Prediction Endpoint

```python
@app.get("/predict")
def predict():
    """
    Main ML prediction endpoint.
    
    Fetches order data, trains a model, and returns predictions
    for each product's demand level.
    
    Returns:
        List of products with their predicted demand level:
        - prediction: 1 = High demand, 0 = Low demand
        - confidence: Model's confidence in the prediction
    """
    # Validate configuration
    if not TOKEN or TOKEN == "":
        raise HTTPException(
            status_code=500, 
            detail="Shopify access token not configured. Please set SHOPIFY_ACCESS_TOKEN in .env"
        )
    
    # Fetch and prepare data
    df = fetch_orders()
    X, y, grouped = prepare_data(df)
    
    # Check if we have enough data
    if len(grouped) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Not enough product data to make predictions. Need at least 2 products with orders."
        )
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions
    grouped["prediction"] = model.predict(X)
    grouped["confidence"] = model.predict_proba(X).max(axis=1).round(3)
    
    # Fetch product names for better readability
    product_names = fetch_products()
    grouped["product_name"] = grouped["product_id"].map(product_names).fillna("Unknown Product")
    
    # Prepare response
    results = grouped[[
        "product_id", 
        "product_name",
        "quantity", 
        "price", 
        "prediction",
        "confidence"
    ]].copy()
    
    results["demand_level"] = results["prediction"].map({1: "High", 0: "Low"})
    results["price"] = results["price"].round(2)
    
    # Sort by prediction confidence
    results = results.sort_values("confidence", ascending=False)
    
    return {
        "total_products": len(results),
        "high_demand_count": int(results["prediction"].sum()),
        "low_demand_count": int((results["prediction"] == 0).sum()),
        "predictions": results.to_dict(orient="records")
    }
```

### Statistics Endpoint

```python
@app.get("/stats")
def get_stats():
    """
    Get basic statistics about order data.
    
    Returns:
        Order and product statistics from the store
    """
    if not TOKEN or TOKEN == "":
        raise HTTPException(
            status_code=500, 
            detail="Shopify access token not configured. Please set SHOPIFY_ACCESS_TOKEN in .env"
        )
    
    df = fetch_orders()
    
    stats = {
        "total_line_items": len(df),
        "unique_products": df["product_id"].nunique(),
        "total_quantity_sold": int(df["quantity"].sum()),
        "avg_price": round(df["price"].mean(), 2),
        "min_price": round(df["price"].min(), 2),
        "max_price": round(df["price"].max(), 2),
    }
    
    # Top selling products
    top_products = df.groupby("product_id")["quantity"].sum().nlargest(5)
    product_names = fetch_products()
    
    stats["top_selling_products"] = [
        {
            "product_id": int(pid),
            "product_name": product_names.get(pid, "Unknown"),
            "total_quantity": int(qty)
        }
        for pid, qty in top_products.items()
    ]
    
    return stats
```

### Run Entry Point

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

## Step 9: Run the App

```bash
uvicorn main:app --reload
```

Open your browser:
- **Root:** http://localhost:8000
- **Predictions:** http://localhost:8000/predict
- **Stats:** http://localhost:8000/stats
- **API Docs:** http://localhost:8000/docs

You now have a **Shopify ML-powered app** 🎉

---

## Example Response

The `/predict` endpoint returns:

```json
{
  "total_products": 15,
  "high_demand_count": 8,
  "low_demand_count": 7,
  "predictions": [
    {
      "product_id": 123456789,
      "product_name": "Amazing T-Shirt",
      "quantity": 45,
      "price": 29.99,
      "prediction": 1,
      "confidence": 0.92,
      "demand_level": "High"
    },
    {
      "product_id": 987654321,
      "product_name": "Basic Mug",
      "quantity": 5,
      "price": 12.99,
      "prediction": 0,
      "confidence": 0.85,
      "demand_level": "Low"
    }
  ]
}
```

- `prediction`: 1 = High demand, 0 = Low demand
- `confidence`: How confident the model is (0.0 to 1.0)

---

## How This Could Be Used in a Real App

- 🏷️ **Tag high-demand products automatically**
- 📈 **Suggest products to promote**
- 🎁 **Power upsells and bundles**
- 📦 **Forecast inventory needs**

---

## Why This App Is Powerful (Even Though It's Simple)

✅ Uses **real Shopify data**  
✅ Introduces **ML without complexity**  
✅ Easy to explain to **clients or recruiters**  
✅ Can be expanded into a **paid app**

---

## Deployment Options

### Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Set environment variables in the dashboard
4. Deploy!

### Railway

1. Create a new project on [Railway](https://railway.app)
2. Deploy from GitHub
3. Add environment variables
4. Done!

---

## What to Build Next

To level this up:

- [ ] **Time-series forecasting** — predict future demand trends
- [ ] **Store data in a database** — track predictions over time
- [ ] **Add a Shopify Admin UI** — embedded app interface
- [ ] **Deploy and connect via App Bridge** — full Shopify integration
- [ ] **Add more features** — seasonality, product categories, customer segments

---

## Complete Code

Here's the full `main.py` for reference:

```python
"""
Smart Product Insights - Shopify ML App
========================================
A simple Shopify app that uses Machine Learning to predict 
whether a product is likely to sell well based on historical order data.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Product Insights",
    description="ML-powered product demand prediction for Shopify stores",
    version="1.0.0"
)

# Shopify Configuration
SHOP = os.getenv("SHOPIFY_STORE_URL", "your-store-name.myshopify.com")
TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
HEADERS = {
    "X-Shopify-Access-Token": TOKEN,
    "Content-Type": "application/json"
}

# Shopify API version
API_VERSION = "2024-01"


def fetch_orders(limit: int = 250) -> pd.DataFrame:
    url = f"https://{SHOP}/admin/api/{API_VERSION}/orders.json?status=any&limit={limit}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch orders from Shopify: {str(e)}")
    
    orders = response.json().get("orders", [])
    
    if not orders:
        raise HTTPException(status_code=404, detail="No orders found in your Shopify store")
    
    rows = []
    for order in orders:
        for item in order.get("line_items", []):
            if item.get("product_id"):
                rows.append({
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "price": float(item["price"])
                })
    
    if not rows:
        raise HTTPException(status_code=404, detail="No product line items found in orders")
    
    return pd.DataFrame(rows)


def fetch_products() -> dict:
    url = f"https://{SHOP}/admin/api/{API_VERSION}/products.json?limit=250"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {}
    
    products = response.json().get("products", [])
    return {p["id"]: p["title"] for p in products}


def prepare_data(df: pd.DataFrame) -> tuple:
    grouped = df.groupby("product_id").agg({
        "quantity": "sum",
        "price": "mean"
    }).reset_index()
    
    median_quantity = grouped["quantity"].median()
    grouped["high_demand"] = (grouped["quantity"] > median_quantity).astype(int)
    
    X = grouped[["quantity", "price"]]
    y = grouped["high_demand"]
    
    return X, y, grouped


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@app.get("/")
def root():
    return {
        "app": "Smart Product Insights",
        "status": "running",
        "endpoints": {
            "/predict": "Get demand predictions for all products",
            "/stats": "Get order statistics",
            "/health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "shop": SHOP}


@app.get("/predict")
def predict():
    if not TOKEN or TOKEN == "":
        raise HTTPException(
            status_code=500, 
            detail="Shopify access token not configured. Please set SHOPIFY_ACCESS_TOKEN in .env"
        )
    
    df = fetch_orders()
    X, y, grouped = prepare_data(df)
    
    if len(grouped) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Not enough product data to make predictions. Need at least 2 products with orders."
        )
    
    model = train_model(X, y)
    
    grouped["prediction"] = model.predict(X)
    grouped["confidence"] = model.predict_proba(X).max(axis=1).round(3)
    
    product_names = fetch_products()
    grouped["product_name"] = grouped["product_id"].map(product_names).fillna("Unknown Product")
    
    results = grouped[[
        "product_id", 
        "product_name",
        "quantity", 
        "price", 
        "prediction",
        "confidence"
    ]].copy()
    
    results["demand_level"] = results["prediction"].map({1: "High", 0: "Low"})
    results["price"] = results["price"].round(2)
    results = results.sort_values("confidence", ascending=False)
    
    return {
        "total_products": len(results),
        "high_demand_count": int(results["prediction"].sum()),
        "low_demand_count": int((results["prediction"] == 0).sum()),
        "predictions": results.to_dict(orient="records")
    }


@app.get("/stats")
def get_stats():
    if not TOKEN or TOKEN == "":
        raise HTTPException(
            status_code=500, 
            detail="Shopify access token not configured. Please set SHOPIFY_ACCESS_TOKEN in .env"
        )
    
    df = fetch_orders()
    
    stats = {
        "total_line_items": len(df),
        "unique_products": df["product_id"].nunique(),
        "total_quantity_sold": int(df["quantity"].sum()),
        "avg_price": round(df["price"].mean(), 2),
        "min_price": round(df["price"].min(), 2),
        "max_price": round(df["price"].max(), 2),
    }
    
    top_products = df.groupby("product_id")["quantity"].sum().nlargest(5)
    product_names = fetch_products()
    
    stats["top_selling_products"] = [
        {
            "product_id": int(pid),
            "product_name": product_names.get(pid, "Unknown"),
            "total_quantity": int(qty)
        }
        for pid, qty in top_products.items()
    ]
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

## Final Thoughts

You don't need advanced AI to build smart Shopify apps.  
Even simple ML models can create **real value** when combined with Shopify data.

If you can build this, you're already ahead of most Shopify developers 🚀

---

## Project Structure

```
shopify-ml-app/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Your Shopify credentials (don't commit!)
├── .gitignore           # Ignore sensitive files
└── README.md            # Project documentation
```

---

*Built with ❤️ using FastAPI and scikit-learn*

