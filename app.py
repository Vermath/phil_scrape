import os
import pandas as pd
import re
import unicodedata
import streamlit as st
import requests
from io import BytesIO
from urllib.parse import urlparse
import logging
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to clean text
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

# Function to fetch content using Jina Reader API
def fetch_content_jina(url, headers=None):
    # Directly append the raw URL without encoding
    jina_read_url = f"https://r.jina.ai/{url}"
    
    try:
        response = requests.get(jina_read_url, headers=headers, timeout=30)
        response.raise_for_status()
        # Assuming the response is in text format
        return response.text if response.text else "n/a"
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred for {url}: {http_err}")
        logger.error(f"Response Content: {http_err.response.text}")
        return "n/a"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return "n/a"

# Streamlit App
def main():
    st.title("🔗 URL Processor and Content Scraper using Jina AI Reader API")

    # Initialize session state for failed URLs
    if 'failed_urls' not in st.session_state:
        st.session_state.failed_urls = []

    # File uploader for CSV
    uploaded_file = st.file_uploader("📁 Upload CSV with 'URL' column", type=["csv"])

    # Manual entry
    st.subheader("🖊️ Or Enter a Single URL")
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
            'Scraped Content': []
        }

        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Optional: Include API key in headers if using a private Jina Reader API instance
        headers = {}
        # Uncomment and modify the following lines if using a private instance with an API key
        # reader_api_key = st.secrets["reader_api"]
        # headers["Authorization"] = f"Bearer {reader_api_key}"

        # Process each URL
        for idx, url in enumerate(urls):
            status_text.text(f"🔄 Processing URL {idx + 1} of {len(urls)}")
            try:
                # Fetch content using Jina Reader API
                scraped_content = fetch_content_jina(url, headers=headers)

                if scraped_content == "n/a":
                    st.warning(f"⚠️ Failed to fetch content for URL: {url}")
                    st.session_state.failed_urls.append(url)

                # Append data
                data['URL'].append(url)
                data['Scraped Content'].append(scraped_content)

            except Exception as e:
                st.warning(f"⚠️ An error occurred while processing URL {url}: {e}")
                data['URL'].append(url)
                data['Scraped Content'].append("n/a")
                st.session_state.failed_urls.append(url)

            # Update progress bar
            progress = (idx + 1) / len(urls)
            progress_bar.progress(progress)

        # Create DataFrame
        df_output = pd.DataFrame(data)

        # Display the updated DataFrame
        st.subheader("📊 Processed Data")
        st.dataframe(df_output)

        # Prepare JSONL for download
        jsonl_lines = df_output.to_json(orient='records', lines=True)
        jsonl_bytes = jsonl_lines.encode('utf-8')
        jsonl_buffer = BytesIO(jsonl_bytes)

        st.download_button(
            label="📥 Download data as JSONL",
            data=jsonl_buffer,
            file_name='scraped_data.jsonl',
            mime='application/json',
        )

        # Also provide CSV download if needed
        csv_buffer = BytesIO()
        df_output.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="📥 Download data as CSV",
            data=csv_buffer,
            file_name='scraped_data.csv',
            mime='text/csv',
        )

        # Display failed URLs if any
        if st.session_state.failed_urls:
            st.subheader("❗ Failed URLs")
            for failed_url in st.session_state.failed_urls:
                st.write(failed_url)

if __name__ == "__main__":
    main()
