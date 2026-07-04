# Log Analysis using Small Language Models (SLM)

## Project Overview

This project focuses on building an offline Log Analysis System using Small Language Models (SLMs). The objective is to create an AI-assisted log analysis pipeline capable of understanding, retrieving, and explaining system logs without relying on internet connectivity.

The project is being developed in multiple phases. This repository currently contains the completed work for **Week 1: Offline Environment Setup**.

---

# Week 1 Objectives

The primary goal of Week 1 was to prepare a fully functional offline development environment that can later be deployed in an isolated environment without internet access.

---

# Tasks Completed

## 1. Python Environment

- Created an isolated Python virtual environment.
- Created an additional clean testing environment to simulate an offline deployment.

Environments used:

- venv
- testenv

---

## 2. Offline Package Preparation

Downloaded all required Python dependencies and stored them inside the `offline_packages` directory for offline installation.

Major packages include:

- torch
- torchvision
- torchaudio
- transformers
- datasets
- peft
- accelerate
- sentence-transformers
- faiss-cpu
- fastapi
- uvicorn
- streamlit
- sentencepiece
- tiktoken

---

## 3. Offline Installation Verification

Verified that all dependencies can be installed without internet using:

```bash
pip install --no-index --find-links ./offline_packages -r requirements.txt