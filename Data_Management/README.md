# 🍲 Recipe Book with AI-Powered Allergen Detection
This university project integrates two distinct datasets to create a comprehensive recipe database enriched with allergen information. The system is designed to support individuals with food allergies by enabling safer, more informed dietary choices.

The project combines recipe data from [TheMealDB API](https://www.themealdb.com/) with allergen data from the "[Food: Allergens and Allergies](https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies)" dataset from Kaggle.

## Core Challenge & Solution: A Dual-Join Approach
A key challenge in this project was accurately mapping recipe ingredients to known allergens, as direct text matching is often insufficient. For instance, an ingredient like "mozzarella" would not be lexically matched to a "Milk" allergy if the allergen dataset only contains the word "cheese."

To solve this, we implemented a dual-integration strategy:

1. Lexical Join: A direct, string-based matching process to link ingredients with allergens for clear, one-to-one cases.

2. Semantic Join: An AI-driven approach using a Sentence Transformer model (all-mpnet-base-v2) to find conceptual relationships. The model generates embeddings for all ingredients and potential allergens, allowing us to identify links based on semantic similarity, successfully connecting terms like "mozzarella" to "cheese."

# Project Workflow
The project is structured into several key stages:

1. Data Acquisition & Preparation
- Acquisition: Data was collected via requests to TheMealDB API and by downloading the allergen dataset from Kaggle.

- Cleaning & Normalization: Both datasets underwent cleaning, including removing null values, converting text to lowercase and trimming whitespace.

- Unit Conversion: Measurement units in the recipe data were standardized to grams (g) and milliliters (ml) to ensure consistency.

2. Data Storage & Modeling in MongoDB
- The entire data ecosystem is managed in MongoDB Atlas.

- A Star Schema was implemented to ensure an efficient and scalable data model:

  - Fact Table: Recipes (containing core recipe information).

  - Dimension Tables: Ingredients and Allergies.

- Pre-aggregated collections were created to optimize query performance for common use cases.

3. Allergen Mapping & Integration
- The lexical and semantic joins were executed to link ingredients to their corresponding allergens.

- The semantic model proved highly effective, overcoming the limitations of the lexical approach and achieving 100% mapping coverage, ensuring every recipe had its potential allergens identified.
