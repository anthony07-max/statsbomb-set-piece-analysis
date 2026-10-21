#!/usr/bin/env bash
set -e
python src/download.py
python src/clean.py
python src/analysis.py
