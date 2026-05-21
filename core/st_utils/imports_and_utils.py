import os
import streamlit as st
import io, zipfile
from core.st_utils.download_video_section import download_video_section
from core.st_utils.sidebar_setting import page_setting
from translations.translations import translate as t

def download_subtitle_zip_button(text: str):
    zip_buffer = io.BytesIO()
    output_dir = "output"
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".srt"):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "rb") as file:
                    zip_file.writestr(file_name, file.read())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label=text,
        data=zip_buffer,
        file_name="subtitles.zip",
        mime="application/zip"
    )

# st.markdown
button_style = """
<style>
/* Global Button Style */
div.stButton > button:first-child {
    display: block;
    padding: 0.5em 1em;
    color: #144070;
    background-color: transparent;
    text-decoration: none;
    font-weight: bold;
    text-align: center;
    transition: all 0.3s ease;
    box-sizing: border-box;
    border: 2px solid #D0DFF2;
    font-size: 1.2em;
}
div.stButton > button:hover {
    background-color: transparent;
    color: #144070;
    border-color: #144070;
}

/* Windows-style Tree Buttons Specificity */
.windows-file-tree div.stButton > button {
    border: none !important;
    background-color: transparent !important;
    text-align: left !important;
    padding: 2px 5px !important;
    font-size: 0.95em !important;
    font-weight: normal !important;
    color: #333 !important;
    justify-content: flex-start !important;
    width: 100% !important;
    min-height: unset !important;
    line-height: 1.2 !important;
    border-radius: 4px !important;
}
.windows-file-tree div.stButton > button:hover {
    background-color: #e8f0fe !important;
    color: #144070 !important;
}

/* Custom Tab Styling - Card Look for Sidebar */
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px !important;
    background-color: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stTabs"] button[data-baseweb="tab"] {
    border: 1px solid #D0DFF2 !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 16px !important;
    background-color: #f0f5ff !important;
    margin-bottom: -1px !important; /* Overlap with panel border */
    z-index: 1 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #ffffff !important;
    border-color: #D0DFF2 !important;
    color: #144070 !important;
    font-weight: bold !important;
    border-top: 3px solid #144070 !important;
}

/* Style the content panel below the tabs */
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] {
    border: 1px solid #D0DFF2 !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px 15px !important;
    background-color: #ffffff !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
}

/* Enhance input visibility within tab panels */
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] div[data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] div[data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] textarea {
    border: 1px solid #c1d5e0 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] div[data-baseweb="input"]:focus-within,
[data-testid="stSidebar"] [data-testid="stTabs"] [data-baseweb="tab-panel"] div[data-baseweb="select"]:focus-within {
    border-color: #144070 !important;
    box-shadow: 0 0 0 1px #144070 !important;
}

/* Hide the default underline */
[data-testid="stSidebar"] [data-testid="stTabs"] [data-testid="stTabHighlight"] {
    display: none !important;
}

div.stDownloadButton > button:first-child {
    display: block;
    padding: 0.5em 1em;
    color: #144070;
    background-color: transparent;
    text-decoration: none;
    font-weight: bold;
    text-align: center;
    transition: all 0.3s ease;
    box-sizing: border-box;
    border: 2px solid #D0DFF2;
    font-size: 1.2em;
}
div.stDownloadButton > button:hover {
    background-color: transparent;
    color: #144070;
    border-color: #144070;
}
div.stDownloadButton > button:active, div.stDownloadButton > button:focus {
    background-color: transparent !important;
    color: #144070 !important;
    border-color: #144070 !important;
    box-shadow: none !important;
}
div.stDownloadButton > button:active:hover, div.stDownloadButton > button:focus:hover {
    background-color: transparent !important;
    color: #144070 !important;
    border-color: #144070 !important;
    box-shadow: none !important;
}

/* Top Header Bar Styling */
.top-header {
    background: linear-gradient(90deg, #144070 0%, #2a6db0 100%);
    padding: 12px 25px;
    border-radius: 10px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 15px rgba(20, 64, 112, 0.15);
}
.top-header-text {
    color: white !important;
    margin: 0 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px;
}

/* Style the very top Streamlit Header bar */
header[data-testid="stHeader"] {
    background-color: #144070 !important;
    border-bottom: 1px solid #0d2b4d !important;
}
/* Adjust color of the status icons and menu in the top bar */
header[data-testid="stHeader"] svg {
    fill: white !important;
}
header[data-testid="stHeader"] div {
    color: white !important;
}

/* 🚀 Framework-level Chinese Localization */
/* 1. Hide Deploy Button */
[data-testid="stHeaderDeploy"] {
    display: none !important;
}

/* 2. Translate "RUNNING..." indicator */
div[data-testid="stStatusWidget"] div[role="img"] + span {
    font-size: 0 !important;
}
div[data-testid="stStatusWidget"] div[role="img"] + span::after {
    content: "正在运行..." !important;
    font-size: 14px !important;
    color: white !important;
}

/* 3. Rebrand Footer */
footer {
    visibility: hidden !important;
}
footer::after {
    content: '© 2026 VideoLingo Q - 专业视频本地化工作站'; 
    visibility: visible !important;
    display: block !important;
    position: relative !important;
    color: #808080 !important;
    padding: 5px !important;
    top: 2px !important;
}

</style>
"""