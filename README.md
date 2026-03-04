# Global Demographic & Economic Clustering App 🌍

An interactive Data Science & Machine Learning application built with **Streamlit** that explores the hidden geopolitical and economic structures of the world. 

By fetching live macroeconomic indicators from the **World Bank API**, this app applies Unsupervised Machine Learning techniques (K-Means and a custom-built Gaussian Mixture Model) to cluster the world. Finally, it uses **Google's Gemini Pro API** to generate intelligent, human-readable insights explaining what defines each global cluster.

## ✨ Features
* **Live World Bank Data Integration:** Dynamically pulls robust demographic and economic data (GDP, Life Expectancy, Literacy grids, CO2 emissions, etc.) with automated handling and KNN imputation for missing values.
* **Custom Machine Learning:** 
  * Baseline **K-Means Clustering** to partition countries.
  * A bespoke **Gaussian Mixture Model (GMM)** implemented from scratch using an Expectation-Maximization loop that enforces a *strict uniform prior*, preventing massive density imbalances and looking for true spatial distributions.
* **Interactive Visualizations:**
  * A vivid, interactive global choropleth map scaling to mathematical clusters using `plotly`.
  * A 2D Principal Component Analysis (PCA) projection scatter plot displaying algorithmic centroids to assess cluster separation.
* **AI-Powered Insights:** Automatically extracts the median statistics of your generated clusters and feeds them into **Gemini 2.5 Pro**, returning a high-level geopolitical executive summary of the global state directly in the app.

---

## 🛠️ Setup & Installation

Follow these instructions to run the application locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/qimiankkk/clustering-countries.git
cd clustering-countries
```

### 2. Create a Virtual Environment (Recommended)
This keeps the project dependencies isolated from your system Python.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set up Google GenAI
To enable the LLM automated insights generated at the bottom of the dashboard, you will need a Google Gemini API Key.
* Get one from [Google AI Studio](https://aistudio.google.com/).
* You can either enter the key directly into the sidebar of the Streamlit app when running, OR set it as an environment variable before launching the app:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 5. Run the Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure
* `app.py`: The main Streamlit dashboard application.
* `data_loader.py`: Handles all World Bank API requests, data cleaning, KNN imputation, log transformation, and standard scaling.
* `models.py`: Contains the clustering algorithms (`run_kmeans` and the custom `custom_gmm_uniform_prior`).
* `visualization.py`: Generates the Plotly choropleth maps and PCA centroid scatter plots.
* `llm_insights.py`: Manages the prompt engineering and connection to the Google GenAI `gemini-2.5-pro` model.
* `test_sanity.py`: A simple CLI script to verify the data and modeling pipeline runs without errors outside of Streamlit.
