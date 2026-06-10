<p align="center">
  <a href="https://openproceedings.org/2026/conf/edbt/paper-334.pdf">
    <img src="https://img.shields.io/badge/paper-EDBT%202026-blue.svg" alt="Paper">
  </a>
  <a href="https://github.com/VasilikiPapanikou/MIMOSA/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
</p>

<p align="center">
  <img src="demo/media/logo.png" width="200">
</p>

<p align="center">
  <em>A Tool for Fairness Exploration Through Explanations</em>
</p>

<p align="center">
  <a href="https://orcid.org/0009-0004-0785-4727">Vasiliki Papanikou</a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0000-0002-3154-6212">Danae Pla Karidi</a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0000-0002-3775-4995">Evaggelia Pitoura</a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0000-0001-9134-9387">Emmanouil Panagiotou</a> &nbsp;·&nbsp;
  <a href="https://orcid.org/0000-0001-5729-1003">Eirini Ntoutsi</a>
</p>

---

As Artificial Intelligence (AI) is increasingly used in areas that impact human lives, concerns about fairness and transparency have grown, especially for protected groups. To better understand such concerns, explainability techniques can be leveraged not only for model interpretation but also to assess potential biases. 
The **MIMOSA**<sup>*</sup> tool utilizes both individual and group explanation methods as bias detectors. It allows users to compare group fairness metrics with explanation findings, identify which features contribute to biased outcomes, visualize explanations through multiple perspectives and apply fairness interventions while tracking how feature contributions change. The tool is designed to be accessible to a wide audience of users, including sociologists, domain experts and machine learning practitioners.

---

## Installation & Usage

### Prerequisites
- Python 3.8+

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py

---
## Citing

If you use MIMOSA in your research, please cite:

```bibtex
@article{papanikou2026mimosa,
  title={MIMOSA: A Tool for Fairness Exploration Through Explanations},
  author={Papanikou, Vasiliki and Karidi, Danae Pla and Pitoura, Evaggelia and Panagiotou, Emmanouil and Ntoutsi, Eirini},
  year={2026}
}
```

<sup>*</sup> *The mimosa flower symbolizes purity, innocence and sensitivity...*
