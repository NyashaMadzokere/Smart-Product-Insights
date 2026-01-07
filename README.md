# 🧠 Smart Product Insights

> A Shopify ML-powered app that predicts product demand using machine learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)

## What Does This App Do?

This app connects to your Shopify store, analyzes historical order data, and uses machine learning to predict which products have **High** or **Low** demand.

**Use cases:**
- 🏷️ Tag high-demand products automatically
- 📈 Suggest products to promote
- 🎁 Power upsells and bundles
- 📦 Forecast inventory needs

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python (FastAPI) |
| ML | scikit-learn (Logistic Regression) |
| Shopify API | Admin REST API |
| Auth | Shopify Admin API token |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- A Shopify store with some order history
- Shopify Admin API access token

### Step 1: Create a Shopify Custom App

1. Go to **Shopify Admin** → **Settings** → **Apps and sales channels**
2. Click **Develop apps** → **Create an app**
3. Name it `Smart Product Insights`
4. Click **Configure Admin API scopes** and enable:
   - ✅ `read_products`
   - ✅ `read_orders`
5. Click **Install app** and copy your **Admin API access token**

### Step 2: Set Up the Project

```bash
# Clone or navigate to the project folder
cd "Shopify ML Project"

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

1. Copy `config.example` to a new file named `.env`
2. Fill in your Shopify credentials:

```env
SHOPIFY_STORE_URL=your-store-name.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_your_actual_token_here
```

### Step 4: Run the App

```bash
# Start the server
uvicorn main:app --reload

# Or run directly
python main.py
```

The app will be available at: **http://localhost:8000**

---

## 📡 API Endpoints

### `GET /`
Health check and list of available endpoints.

### `GET /predict`
**Main ML endpoint** - Returns demand predictions for all products.

**Response:**
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
    }
  ]
}
```

### `GET /stats`
Get order statistics from your store.

**Response:**
```json
{
  "total_line_items": 1250,
  "unique_products": 45,
  "total_quantity_sold": 3200,
  "avg_price": 34.50,
  "top_selling_products": [...]
}
```

### `GET /health`
Simple health check endpoint.

---

## 🤖 How the ML Model Works

1. **Data Collection**: Fetches all orders from your Shopify store
2. **Feature Engineering**: Groups by product and calculates:
   - Total quantity sold
   - Average price
3. **Labeling**: Products are labeled based on median quantity:
   - Above median → **High demand** (1)
   - Below median → **Low demand** (0)
4. **Training**: Logistic Regression model learns patterns
5. **Prediction**: Model predicts demand level with confidence score

---

## 🛠️ Deployment Options

### Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Render
1. Create a new Web Service on [Render](https://render.com)
2. Connect your repository
3. Set environment variables in Render dashboard
4. Deploy!

### Railway
1. Create a new project on [Railway](https://railway.app)
2. Deploy from GitHub
3. Add environment variables
4. Done!

---

## 📈 What to Build Next

To level up this app:

- [ ] **Time-series forecasting** - Predict future demand trends
- [ ] **Database storage** - Store predictions history
- [ ] **Shopify Admin UI** - Build an embedded app interface
- [ ] **Webhooks** - Real-time updates on new orders
- [ ] **More features** - Include product tags, categories, seasonality

---

## 📁 Project Structure

```
shopify-ml-app/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── config.example       # Example environment config
├── .env                 # Your actual config (create this)
└── README.md            # This file
```

---

## 🐛 Troubleshooting

### "No orders found"
- Make sure your store has order history
- Check that your API token has `read_orders` permission

### "Failed to fetch orders"
- Verify your `SHOPIFY_STORE_URL` is correct (no `https://`)
- Check your `SHOPIFY_ACCESS_TOKEN` is valid

### "Not enough product data"
- You need at least 2 products with orders for predictions
- Try generating some test orders in your development store

---

## 📄 License

MIT License - feel free to use this for your own projects!


