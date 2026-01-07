"""
Smart Product Insights - Shopify ML App
========================================
A simple Shopify app that uses Machine Learning to predict 
whether a product is likely to sell well  based on historical order data.
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
SHOP = os.getenv("SHOPIFY_STORE_URL", "XXXXXXXXXXXXX.myshopify.com")
TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
HEADERS = {
    "X-Shopify-Access-Token": TOKEN,
    "Content-Type": "application/json"
}

# Shopify API version
API_VERSION = "2025-01"


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


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "shop": SHOP}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

