Graph Analysis Toolkit

This project is part of CECS 427 — Social and Large-Scale Networks.
It provides a flexible Python application for analyzing, partitioning, and visualizing graph structures with support for advanced features like clustering coefficients, neighborhood overlap, homophily testing, structural balance, temporal dynamics, and robustness under failures.

🚀 Features
Load & Save Graphs from .gml files.
Clustering Coefficient visualization (node size = CC, color = degree).
Neighborhood Overlap computation (edge thickness & color).
Community Detection using Girvan–Newman.
Homophily Test (one-tailed Z-test / t-test).
Structural Balance verification on signed graphs.
Simulate Failures by removing random edges and analyzing robustness.
Temporal Simulation of evolving networks from CSV input.
Export enriched graphs back to .gml.

Setup
1. Clone the repo: git clone https://github.com/YOUR_USERNAME/graph-analysis.git
cd graph-analysis
2. Create a Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Mac/Linux

3.  Install Dependencies
   pip install -r requirements.txt
# or: pip install networkx matplotlib scipy
# 4) run your script
python graph_analysis.py sample_full.gml --plot T --temporal_simulation edges.csv



📂 Repo Structure
graph-analysis/
│── graph_analysis.py    # Main script
│── requirements.txt     # Dependencies
│── sample.gml           # Example graph
│── edges.csv            # Example temporal changes
│── plots/               # Auto-generated visualizations
│── parts/               # Exported communities
│── README.md            # Documentation
│── .gitignore           # Git ignore rules

👨‍💻 Author
Santosh Adhikari (Tony)
California State University, Long Beach — Fall 2025 (CECS 427)
