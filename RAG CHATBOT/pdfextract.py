import streamlit as st
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import pdfkit

# Function to scrape content from URL
def scrape_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        return text_content
    else:
        return None

# Function to save content to PDF using FPDF
def save_content_to_pdf_fpdf(content, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Handle Unicode characters
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.output(pdf_path)

# Function to save content to PDF using pdfkit
def save_content_to_pdf_pdfkit(content, pdf_path):
    # Write content to an HTML file
    html_path = "content.html"
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(f"<html><body><pre>{content}</pre></body></html>")
    
    # Path to wkhtmltopdf executable
    path_to_wkhtmltopdf = r"C:\Users\yaswa\Downloads\wkhtmltopdf-0.12.6-installer.exe"  # Change this to your installation path
    
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    # Convert HTML file to PDF
    pdfkit.from_file(html_path, pdf_path, configuration=config)

# Streamlit app
def main():
    st.set_page_config(page_title="Web Scraper and PDF Converter")
    st.title("Web Scraper and PDF Converter")

    url = st.text_input("Enter the URL to scrape:")
    if st.button("Scrape and Generate PDF"):
        if url:
            with st.spinner("Scraping content..."):
                content = scrape_content(url)
            
            if content:
                st.success("Content scraped successfully!")
                st.text_area("Scraped Content", content, height=300)

                # Save to PDF using FPDF
                pdf_path_fpdf = "output_fpdf.pdf"
                save_content_to_pdf_fpdf(content, pdf_path_fpdf)

                # Save to PDF using pdfkit
                pdf_path_pdfkit = "output_pdfkit.pdf"
                save_content_to_pdf_pdfkit(content, pdf_path_pdfkit)

                # Provide download links
                with open(pdf_path_fpdf, "rb") as pdf_file_fpdf:
                    st.download_button("Download PDF (FPDF)", pdf_file_fpdf, file_name="output_fpdf.pdf")

                with open(pdf_path_pdfkit, "rb") as pdf_file_pdfkit:
                    st.download_button("Download PDF (pdfkit)", pdf_file_pdfkit, file_name="output_pdfkit.pdf")
            else:
                st.error("Failed to scrape content from the provided URL.")
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
