# Continuum Shape–Force Estimation (PCK) — RAL 2025

This repository provides the source scripts accompanying our IEEE **Robotics and Automation Letters (RAL) 2025** paper on  
**Polynomial Curvature (PCK)–based Shape–Force Estimation** for continuum robots.

It contains the minimal Python implementation used for generating simulation and visualization results presented in the manuscript.

---

## 📦 Repository structure
```
continuum-shapeforce-pck-ral2025/
├─ scripts/
│  ├─ pck_shape_estimation_testplot_2D.py
│  ├─ pck_shape_estimation_testplot_6D.py
│  └─ plot_shape_force_corr_2x2.py
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## ⚙️ Environment setup (Windows)
Open **Command Prompt** or **PowerShell** in this folder:

```cmd
:: (optional) create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

:: install dependencies
pip install -r requirements.txt
```

If you already have `numpy`, `scipy`, `matplotlib`, and `pandas` installed, you can skip the virtual environment step.

---

## ▶️ Run examples

All scripts can be executed directly from the terminal (**PowerShell** or **Command Prompt**).  
We recommend running inside a virtual environment (`.venv`) after installing dependencies.

### 1️⃣ Planar shape–force estimation (PCK0/1/2)
```powershell
python scripts\pck_shape_estimation_testplot_2D.py `
  --grid-mode compass3 `
  --Fgrid "[[1.0,1.0]]" `
  --tensions "[[2,2,0,0],[4,4,0,0]]" `
  --shape-table --force-table `
  --shape-overall-mae --shape-overall-mae-path shape_overall_mae.csv
```

**Generates:**
- `shape_table.csv` – shape estimation results  
- `force_table.csv` – estimated vs. ground-truth forces  
- `shape_overall_mae.csv` – mean absolute error summary  
- a plot window (and optional saved PNG)

---

### 2️⃣ 6D solver variant (base-frame wrench estimation)
```powershell
python scripts\pck_shape_estimation_testplot_6D.py `
  --solver full6d `
  --shape-table --force-table
```

**Generates:**
- 6D wrench estimation results (tip and base)  
- comparison plots and CSVs  

---

### 3️⃣ Error-coupling visualization (shape–force correlation)
```powershell
python scripts\plot_shape_force_corr_2x2.py `
  --shape shape_table.csv `
  --force force_table.csv
```

**Generates:**
- 2×2 correlation plot (PNG or pop-up window)  
- optional table of Pearson correlation coefficients  

---

### Notes
- Default parameters reproduce the configurations used in the **RAL 2025 manuscript**.  
- For custom tension profiles or force grids, modify `--tensions` and `--Fgrid`.  
- All outputs are saved to the **repository root directory** unless otherwise specified.  
- Figures and CSVs can be reused or adapted under the MIT License.  

---

## 📁 Script summary

| Script | Description |
|---------|--------------|
| `pck_shape_estimation_testplot_2D.py` | Planar PCK0/1/2 estimation and result export. |
| `pck_shape_estimation_testplot_6D.py` | Extended solver for full 6D wrench estimation. |
| `plot_shape_force_corr_2x2.py` | Produces shape–force error-coupling correlation plots. |

---

## 🧩 Expected results

Running all three commands will reproduce:
| Output | Description |
|---------|-------------|
| **CSV files** | Estimated shape and force data (used in paper tables). |
| **Plots** | Shape reconstruction and 2×2 error-coupling figures. |
| **Metrics** | Mean absolute error summaries for PCK0/1/2. |

All results correspond to the planar and 6D examples discussed in the RAL manuscript.

---

## 🪪 License
This code is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.

---

## 🧭 Citation
If you use this code, please cite:

> Guoqing Zhang *et al.*,  
> “Integrated Shape–Force Estimation for Continuum Robots: A Virtual-Work and Polynomial-Curvature Framework,”  
> *IEEE Robotics and Automation Letters (RAL)*, 2025.

---

## 📚 Requirements
```
numpy
scipy
matplotlib
pandas
```

Tested on Python 3.10+ (Windows 10).  
No additional dependencies are required.
