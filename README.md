# IroAdvisor <sup>[Beta]
Robo Advisor algorithm design for drawdown-based optimization of investment portfolios.

## Table of contents
1. [Team](#team)
2. [Partner](#partner)
3. [Installation](#api-wrappers)
4. [Project Organization](#project-organization)
5. [Conceptual Slides](#conceptual-slides)
5. [Technical Memo](#technical-memo)


## Team
This project has been carried out by 4th year students of the Degree in Data Science and Engineering of the University Carlos III de Madrid (UC3M) within the framework of the subject Data Science Project in Madrid, Spain, January 2022.

* **Bernardo Bouzas** ([Linkedin](https://www.linkedin.com/in/bbouzas/)) ([Twitter](https://www.twitter.com/berniBZS/)) ([Github](https://github.com/berniBZS))
* **David Méndez** ([Linkedin](https://www.linkedin.com/in/david-m%C3%A9ndez-encinas-4313221b4/)) ([Github](https://github.com/berniBZS))
* **Claudio Sotillos** ([Linkedin](https://www.linkedin.com/in/claudio-sotillos-peceroso-a1a240217/)) ([Github](https://github.com/berniBZS))
* **Laura Torregrosa**

## Partner
This project has been possible thanks to the collaboration of **IronIA Fintech** (SIMPLICITAS CAPITAL, S.L), a financial services provider in Spain that makes available to retail investors tens of thousands of investment funds.

IronIA was faced with the need to build a Robo Advisor to suggest fund investment portfolios in an optimal and personalized way, and the team has designed an end-to-end solution in order to successfully solve their problem. 



## Installation
Create a **Python 3.9** virtual environment and run the following command:
```
pip install -r requirements.txt
```

## Project organization

| Module                            | Description                                                              |
|-----------------------------------|--------------------------------------------------------------------------|
| **.streamlit / app.py**              | Contains all the code needed to deploy an MVP with the streamlit SDK.                                   |
| **data**                   | All the data extracted and refined needed to develop tests and solutions.                                                              |
| **market characterization**                  | Raw data, scrapers and refined data for obtaining +40 market-characterizing variables.                                                           |
| **notebooks**    | Jupyter Notebooks containing initial data cleaning and  displays, as well as market-characterizing variables clustering insights.                                              |
| **figures**    | Clustering and test results relevant figures.                                              |
| **src**               | Main directory containing all python algorithms used to implement the full optimization pipeline.                                                         |
| **reports**               | Folder containing PDF files: technical memo and presentation slides.                                                         |
------------------------------------------------------------

## Conceptual slides 

A set of slides is also attached, which covers in a light and storytelling way the main challenges faced and the process behind every implementation.

This presentation was defended in the Degrees Hall of the Polytechnic School of Engineering of the UC3M in Leganés. The defense can be seen at (link not yet available)

## Technical Memo

A 48-page technical report is also attached, which exhaustively details the approach to the problem, the methodology used, implemented algorithms, mathematical foundation, socio-economic context and in-depth description of each element of this project.

Said document covers the following contents:

###Introduction
* Introduction to the partner 
* Introduction to the project
###Initial thoughts
* Historical background
* Available data overview
* Working in a cloud environment
* Prices dataframe
* Categories dataframe
* Ratios dataframe
* Overcoming main dataset challenges
###Initial portfolio allocation
* Linear vs nonlinear programming algorithms
###Risk management optimization
* Conditional Value-At-Risk
* Conditional Drawdown-at-risk
* Mean-Absolute Deviation
* Maximum Loss
* Market Neutrality
* Inherited assumptions
* Core approach
* Linearization
###Technical methodology
* pyportfolioopt library
* pyomo library
* Experiment
* Results 
###Phase 2: Model refinement
* Fund classes adjustment
* Hierarchical computation
* Betas challenges
* Risk-balanced portfolio adjustment
###Multi-characterization of global markets 
* Introduction and key ideas
* Market evaluation
* Variables selection
* Scraping methodology
* Data preprocessing and tuning
###Proximity-based portfolio assimilation 
* Clustering: 1st approach
* Distance matching: 2nd approach 
* Comparing approaches
###IroAdvisor_v1.01: Minimum Viable Product 
* Aim of the MVP
* Risk aversion assessment
* Tool deployment
###Final results
###Future improvements
###Final conclusions
###References
###Extra bibliography consulted