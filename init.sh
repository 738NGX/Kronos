#!/bin/bash
pip install -r requirements.txt
apt update && apt install fonts-noto-cjk

# Download and extract Qlib data
# wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
# mkdir -p ~/.qlib/qlib_data/cn_data
# tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
# rm -f qlib_bin.tar.gz