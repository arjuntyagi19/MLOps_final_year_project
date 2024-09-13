# SMILES-Generation: Comparative Analysis of GANs, VAEs, and Graph Neural Networks

This repository contains the implementation of a comparative analysis of Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Graph Neural Networks (GNNs) for generating valid SMILES (Simplified Molecular Input Line Entry System) strings. These SMILES strings represent molecular structures and are used in drug discovery and material science applications. A web application will also be developed to explore and visualize the results of each model.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Comparative Analysis](#comparative-analysis)
- [MLOps Workflow](#mlops-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to compare the performance of three machine learning models—GANs, VAEs, and Graph Neural Networks (GNNs)—in generating valid SMILES strings. The generated SMILES are applied to real-world problems in drug discovery and material science. Additionally, a web application will be developed to provide a user-friendly interface for researchers to explore and analyze the generated molecular structures.

## Problem Statement
Traditional methods for generating SMILES strings, such as combinatorial algorithms, are computationally expensive and fail to handle the complexity and diversity of molecular structures needed for practical applications. In this project, we aim to compare three machine learning approaches—GANs, VAEs, and GNNs—to determine which is the most efficient for generating diverse and valid SMILES strings.

## Project Architecture
The project consists of the following components:
1. **Data Ingestion**: Collect and preprocess molecular datasets for training.
2. **Model Development**: Implement GAN, VAE, and GNN models for SMILES generation.
3. **Model Evaluation**: Evaluate and compare the generated SMILES strings from each model.
4. **Deployment**: Deploy the models and results to a web application.
5. **Web Application**: A Streamlit-based interface for users to interact with and visualize the generated SMILES strings.

## Comparative Analysis
We will compare the performance of the following models:
- **GANs (Generative Adversarial Networks)**: Generate SMILES strings through adversarial training.
- **VAEs (Variational Autoencoders)**: Learn a latent representation of molecular structures and generate SMILES strings.
- **GNNs (Graph Neural Networks)**: Model molecular structures as graphs and generate SMILES strings based on learned graph representations.

### Evaluation Metrics:
- **Validity**: The percentage of generated SMILES strings that represent valid chemical molecules.
- **Uniqueness**: The proportion of unique SMILES strings generated.
- **Novelty**: The proportion of generated molecules that are not present in the training dataset.
- **Property-based Metrics**: Chemical properties like molecular weight, solubility, and bioactivity of the generated molecules.

## MLOps Workflow
The project follows a cloud-compatible MLOps pipeline for GANs, VAEs, and GNNs.

1. **Data Ingestion**: Store datasets in cloud storage (AWS S3, Azure Blob, or GCP Storage).
2. **Model Training**: Train each model on GPU-enabled cloud environments (EC2, Azure ML, GCP AI).
3. **Model Evaluation**: Validate the generated SMILES using RDKit and compare the models based on defined metrics.
4. **Model Packaging**: Containerize each model using Docker.
5. **Deployment**: Deploy models via AWS Elastic Beanstalk, Azure App Services, or Google App Engine.
6. **CI/CD**: Set up continuous integration and deployment pipelines for all models.

## Installation

### Requirements
- Python 3.8+
- PyTorch or TensorFlow
- RDKit
- Streamlit
- Docker (for containerization)
- Cloud CLI (AWS CLI, Azure CLI, or Google Cloud SDK)


