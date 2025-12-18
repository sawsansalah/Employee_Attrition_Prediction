# Employee_Attrition_Prediction
# 📦 DVC Employee Attrition Prediction

This project demonstrates how to use **DVC (Data Version Control)** with **Git**, **Python**, and **AWS S3** to build a **reproducible, real-life machine learning pipeline**.

The use case is **Employee Attrition Prediction**, a common business problem where HR teams want to predict whether an employee is likely to leave the company.

---

## 🎯 Project Goals

* Version control **datasets** and **models** using DVC
* Store large files in **AWS S3**, not Git
* Reproduce ML pipelines easily across teams and environments
* Learn a real-world **MLOps workflow**

---

## 🏢 Business Use Case

HR wants to predict employee attrition based on:

* Age
* Monthly income
* Years at company
* Overtime

The trained model helps HR:

* Reduce attrition
* Plan hiring
* Improve employee retention strategies

---

## 📂 Project Structure

```text
.
├── data/
│   └── employee_data.csv      # Tracked by DVC
├── models/
│   └── attrition_model.pkl    # Generated model (DVC output)
├── train.py                   # Training script
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # Exact pipeline state
├── .dvc/config                # DVC remote (S3) configuration
├── .gitignore
└── README.md
```

---

## 🧾 Dataset Description

**File:** `data/employee_data.csv`

| Column           | Description          |
| ---------------- | -------------------- |
| age              | Employee age         |
| monthly_income   | Monthly salary       |
| years_at_company | Years in company     |
| overtime         | 1 = Yes, 0 = No      |
| left_company     | 1 = Left, 0 = Stayed |

---

## 🛠️ Tech Stack

* Python 3.10+
* scikit-learn
* DVC
* Git
* AWS S3

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <repo-url>
cd employee-attrition-dvc
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
pip install "dvc[s3]"
```

---

### 3️⃣ Configure AWS credentials

```bash
aws configure
```

> Make sure your IAM user has access to the S3 bucket used by DVC.

---

## ☁️ DVC Remote (S3)

The project uses **AWS S3** as a DVC remote to store datasets and trained models.

Example configuration:

```bash
dvc remote add -d s3remote s3://company-ml-dvc-storage/employee-attrition
```

This configuration is stored in:

```text
.dvc/config
```

---

## 📦 Tracking Data with DVC

```bash
dvc add data/employee_data.csv
git add data/employee_data.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

---

## 🔁 Running the ML Pipeline

Create the pipeline:

```bash
dvc stage add \
  -n train \
  -d train.py \
  -d data/employee_data.csv \
  -o models/attrition_model.pkl \
  python train.py
```

Run the pipeline:

```bash
dvc repro
```

---

## ⬆️ Push Artifacts to S3

```bash
dvc push
```

Uploads:

* Dataset
* Trained model

Git only stores **metadata**, not large files.

---

## ⬇️ Pull Artifacts from S3

For teammates or CI/CD:

```bash
dvc pull
```

This restores:

* Exact dataset version
* Exact trained model

---

## 🔄 Typical Workflow

1. Update dataset
2. Run `dvc add`
3. Run `dvc repro`
4. Run `dvc push`
5. Commit & push Git changes

---

## ✅ Best Practices

* ❌ Do not commit AWS credentials
* ✅ Use IAM roles when possible
* ✅ Separate dev/prod DVC remotes
* ✅ Keep data and models out of Git

---

## 📌 What You Learn from This Project

* Real-life DVC usage
* Data & model versioning
* Reproducible ML pipelines
* Foundation for CI/CD and MLOps

---

## 📈 Next Improvements

* Add `params.yaml`
* Add metrics tracking
* Integrate GitHub Actions
* Add model evaluation reports

---

## 👤 Author

Created for learning **DVC and MLOps best practices**.

---

