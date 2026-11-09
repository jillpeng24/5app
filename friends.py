"""
朋友版的入口：直接執行同一份 5-nokey.py。
Streamlit 不允許同一個檔案部署兩次，所以多一個入口檔。
這個 App 的 Secrets 不放永豐金鑰 → 自動改用 Yahoo（延遲報價）。
以後只要改 5-nokey.py，兩個網址會一起更新，這個檔案不用動。
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "5-nokey.py"), run_name="__main__")
