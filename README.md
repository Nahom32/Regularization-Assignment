# README: Regularization via Noise Injection in Classification

## **Introduction**
This project addresses the issue of overfitting in classification models by implementing a regularization technique called **noise injection**. Overfitting occurs when a model learns specific patterns and noise from the training data, leading to poor generalization on unseen data. Using the **Star dataset**, a Decision Tree classifier is conditioned to overfit, and regularization is applied by injecting Gaussian noise into the training data. The project evaluates the performance of the overfitted and regularized models using metrics and visualizations.

---

## **Objective**
The primary goal of this project is to:
1. Train a Decision Tree classifier to demonstrate overfitting.
2. Regularize the overfitted model using noise injection.
3. Compare the performance of the overfitted and regularized models using training and testing accuracy metrics.
4. Visualize the effects of regularization using decision boundaries, accuracy charts, and other evaluation metrics.

---

## **Dataset**
The dataset used in this project is the **Star dataset**, which contains features and classifications of various types of stars. The target variable for classification is **Star Category**, and irrelevant columns such as "Star Type," "Spectral Class," and "Star Color" were excluded from the feature set.

### Dataset Structure
- **Features:** Numerical columns representing properties of stars.
- **Target:** Star Category (e.g., Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant, Hypergiant).

---

## **Methodology**
### 1. **Data Preprocessing:**
   - Selected relevant features and excluded unnecessary columns.
   - Split the data into training and testing sets (80% training, 20% testing).

### 2. **Overfitted Model:**
   - Trained a Decision Tree classifier with no constraints on depth (`max_depth=None`).
   - Recorded training and testing accuracies to demonstrate overfitting.

### 3. **Regularized Model:**
   - Applied Gaussian noise to the training data (mean=0, standard deviation=0.1).
   - Retrained the Decision Tree classifier on the noisy dataset.
   - Evaluated the regularized model's performance on training and testing data.

### 4. **Evaluation:**
   - Compared the training and testing accuracies of the overfitted and regularized models.
   - Visualized the performance using bar charts and decision boundaries.

---

## **Results**
### Key Observations:
1. **Overfitted Model:**
   - Achieved near-perfect accuracy on the training data.
   - Performed poorly on testing data, indicating a lack of generalization.

2. **Regularized Model:**
   - Training accuracy decreased slightly due to noise injection.
   - Testing accuracy improved significantly, showcasing better generalization.

### Visualizations:
- **Bar Charts:** Highlighted the reduced gap between training and testing accuracies after regularization.
- **Decision Boundaries:** Demonstrated smoother, more generalized boundaries in the regularized model compared to the overfitted model.

---

## **Usage**
### Requirements:
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`

### Running the Code:
1. Clone the repository.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. View the visualizations and accuracy results in the console and generated plots.

---

## **Conclusion**
Noise injection is an effective regularization technique for reducing overfitting in high-capacity models like Decision Trees. By introducing small perturbations to the training data, the model is encouraged to generalize better and focus on broader trends. This project demonstrates the utility of noise injection in improving the generalization of an overfitted model while maintaining reasonable accuracy.

## **Group Members**
| Name                    | ID         |
|-------------------------|------------|
| Nahom Senay             | GSR/4848/17| 
| Tigist Wondimagegnehu   | GSR/5506/17|
