import os
import pandas as pd
import re
import unicodedata
import streamlit as st
from crawl4ai import WebCrawler
from crawl4ai.crawler_strategy import LocalSeleniumCrawlerStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from io import BytesIO
from urllib.parse import urlparse
import logging
import subprocess
import shutil
import json
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to verify Chromium installation
def verify_chromium():
    st.subheader("🔍 Chromium Installation Verification")
    try:
        result = subprocess.run(['chromium', '--version'], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        st.code(version)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"🔴 Chromium not found or error occurred: {e}")
        return False
    except FileNotFoundError:
        st.error("🔴 Chromium binary not found.")
        return False

# Function to verify Chromedriver installation
def verify_chromedriver():
    st.subheader("🔍 Chromedriver Installation Verification")
    chromedriver_path = shutil.which("chromedriver")
    if chromedriver_path:
        try:
            result = subprocess.run(['chromedriver', '--version'], capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            st.code(version)
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"🔴 Chromedriver error: {e}")
            return False
    else:
        st.error("🔴 Chromedriver binary not found.")
        return False

# Utility function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'\(TM\)', '', text)
    text = re.sub(r'â€', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to parse pasted URLs
def parse_pasted_urls(urls_text):
    urls = re.split(r'[,\n\s]+', urls_text)
    return [url.strip() for url in urls if url.strip()]

# Function to validate URLs
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Function to configure Selenium with enhanced options
def configure_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Use headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Spoof a common user-agent to mimic a real browser
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
    chrome_options.add_argument(f"user-agent={user_agent}")
    
    # Disable images and CSS to speed up scraping and reduce detection
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.javascript": 1,  # Enable JS
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Additional stealth options can be added here if needed
    
    chrome_options.binary_location = shutil.which("chromium")  # Automatically find the chromium binary
    
    return chrome_options

# Function to define LLM Extraction Strategy with refined instructions
def get_llm_extraction_strategy(api_key):
    return LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",  # Ensure this model is available and supported
        api_token=api_key,  # Use the OpenAI API key from secrets
        instruction=(
            "Please extract only the main article content from the following webpage. "
            "Exclude any navigation menus, headers, footers, advertisements, or any non-essential elements. "
            "Provide a clear and concise summary of the main content without including any HTML or markup."
        )
        # Optionally, you can add a schema or other parameters as needed
    )

# Streamlit App
def main():
    st.title("🔗 URL Processor and Content Extractor with LLM")

    # Verify Chromium and Chromedriver installation
    chromium_ok = verify_chromium()
    chromedriver_ok = verify_chromedriver()
    if not (chromium_ok and chromedriver_ok):
        st.error("🔴 Chromium or Chromedriver is not installed correctly. Please check your setup.")
        st.stop()

    # Configure Selenium
    chrome_options = configure_selenium()

    # Locate system-installed Chromedriver
    chromedriver_path = shutil.which("chromedriver")
    if not chromedriver_path:
        st.error("🔴 Chromedriver not found in system PATH.")
        st.stop()

    # Initialize WebDriver with enhanced options
    try:
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        crawler_strategy = LocalSeleniumCrawlerStrategy(driver=driver)
        crawler = WebCrawler(verbose=False, crawler_strategy=crawler_strategy)
        crawler.warmup()
        st.success("🟢 Selenium WebDriver initialized successfully.")
    except Exception as e:
        st.error(f"🔴 Selenium initialization error: {e}")
        st.stop()

    # Initialize session state for failed URLs
    if 'failed_urls' not in st.session_state:
        st.session_state.failed_urls = []

    # File uploader for CSV
    uploaded_file = st.file_uploader("📁 Upload CSV with 'URL' column", type=["csv"])

    # Manual entry
    st.subheader("🖊️ Or Enter URLs Manually")
    manual_url = st.text_input("Enter a single URL")

    # Paste list of URLs
    st.subheader("📋 Or Paste a List of URLs")
    pasted_urls = st.text_area("Paste your URLs here (separated by commas, newlines, or spaces)")

    # Button to start processing
    if st.button("🚀 Process URLs"):
        urls = []

        # Handle uploaded CSV
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                if 'URL' not in df_input.columns:
                    st.error("❌ CSV file must contain a 'URL' column.")
                else:
                    uploaded_urls = df_input['URL'].dropna().tolist()
                    # Validate URLs
                    valid_uploaded_urls = [url for url in uploaded_urls if is_valid_url(url)]
                    invalid_uploaded_urls = [url for url in uploaded_urls if not is_valid_url(url)]
                    urls.extend(valid_uploaded_urls)
                    st.success(f"✅ Loaded {len(valid_uploaded_urls)} valid URLs from the uploaded CSV.")
                    if invalid_uploaded_urls:
                        st.warning(f"⚠️ {len(invalid_uploaded_urls)} invalid URLs were skipped from the uploaded CSV.")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")

        # Handle manual entry
        if manual_url:
            if is_valid_url(manual_url):
                urls.append(manual_url)
                st.success("✅ Added manually entered URL.")
            else:
                st.warning("⚠️ The manually entered URL is invalid and was skipped.")

        # Handle pasted URLs
        if pasted_urls:
            parsed_urls = parse_pasted_urls(pasted_urls)
            valid_pasted_urls = [url for url in parsed_urls if is_valid_url(url)]
            invalid_pasted_urls = [url for url in parsed_urls if not is_valid_url(url)]
            urls.extend(valid_pasted_urls)
            st.success(f"✅ Added {len(valid_pasted_urls)} valid URLs from pasted list.")
            if invalid_pasted_urls:
                st.warning(f"⚠️ {len(invalid_pasted_urls)} invalid URLs were skipped from the pasted list.")

        if not urls:
            st.error("❌ No valid URLs provided. Please upload a CSV, enter URLs manually, or paste a list of URLs.")
            return

        # Remove duplicates
        urls = list(dict.fromkeys(urls))
        st.write(f"📊 **Total unique valid URLs to process:** {len(urls)}")

        # Initialize lists for DataFrame
        data = {
            'URL': [],
            'Extracted Content': []
        }

        # Define the LLM Extraction Strategy
        llm_extraction_strategy = get_llm_extraction_strategy(st.secrets["openai_api_key"])

        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process each URL with enhanced scraping strategies
        for idx, url in enumerate(urls):
            status_text.text(f"🔄 Processing URL {idx + 1} of {len(urls)}")
            try:
                # Introduce random delay to mimic human behavior
                time.sleep(random.uniform(1, 3))

                # Scrape the webpage with LLM Extraction Strategy
                scrape_result = crawler.run(
                    url=url,
                    extraction_strategy=llm_extraction_strategy,
                    bypass_cache=True
                )

                if scrape_result.success:
                    extracted_content = scrape_result.extracted_content
                    if not extracted_content:
                        st.warning(f"⚠️ Extracted content is empty for URL: {url}")
                        extracted_content = "n/a"
                    else:
                        extracted_content = clean_text(extracted_content)
                else:
                    st.warning(f"⚠️ Failed to extract content from the URL: {url}")
                    extracted_content = "n/a"
                    st.session_state.failed_urls.append(url)

                # Append data
                data['URL'].append(url)
                data['Extracted Content'].append(extracted_content if extracted_content else "n/a")

            except Exception as e:
                st.warning(f"⚠️ An error occurred while processing URL {url}: {e}")
                data['URL'].append(url)
                data['Extracted Content'].append("n/a")
                st.session_state.failed_urls.append(url)

            # Update progress bar
            progress = (idx + 1) / len(urls)
            progress_bar.progress(progress)

        # Create DataFrame
        df_output = pd.DataFrame(data)

        # Display the updated DataFrame with expandable markdown content
        st.subheader("📊 Extracted Data")

        # Function to display markdown content in an expandable section
        def display_markdown_table(df):
            for index, row in df.iterrows():
                with st.expander(f"URL: {row['URL']}"):
                    st.markdown(row['Extracted Content'])

        display_markdown_table(df_output)

        # Prepare JSONL for download
        jsonl_lines = df_output.to_json(orient='records', lines=True)
        jsonl_bytes = jsonl_lines.encode('utf-8')
        jsonl_buffer = BytesIO(jsonl_bytes)

        st.download_button(
            label="📥 Download data as JSONL",
            data=jsonl_buffer,
            file_name='extracted_data.jsonl',
            mime='application/json',
        )

        # Also provide CSV download if needed
        csv_buffer = BytesIO()
        df_output.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="📥 Download data as CSV",
            data=csv_buffer,
            file_name='extracted_data.csv',
            mime='text/csv',
        )

        # Display failed URLs if any
        if st.session_state.failed_urls:
            st.subheader("❗ Failed URLs")
            for failed_url in st.session_state.failed_urls:
                st.write(failed_url)

if __name__ == "__main__":
    main()
