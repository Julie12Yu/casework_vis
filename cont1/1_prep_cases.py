import os
import pickle
from pathlib import Path
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import PyPDF2
from openai import OpenAI
import time

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DRIVE_FOLDER_NAME = "nov_12_court_pdfs"
OUTPUT_FILE = "court_cases_with_summaries.json"

# Maximum number of pages to extract from each PDF
# 20 pages is typically enough for court opinions (covers intro, facts, analysis, holding)
# Adjust this if you want more or fewer pages
MAX_PAGES = 17

# Number of last pages to extract (for conclusions/outcomes)
# Last 3 pages typically contain the holding, decision, and any final orders
LAST_PAGES = 5

def authenticate_google_drive():
    """Authenticate with Google Drive API"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def find_folder_id(service, folder_name):
    """Find folder ID by name"""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        pageSize=10
    ).execute()
    
    folders = results.get('files', [])
    
    return folders[0]['id']

def download_pdfs_from_drive(service, folder_id=None, max_pages=20, last_pages=3):
    query = "mimeType='application/pdf'"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    
    # Handle pagination to get all files
    files = []
    page_token = None
    
    while True:
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token
        ).execute()
        
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        
        if not page_token:
            break
    
    print(f"Found {len(files)} PDF files in folder")
    
    documents = []
    for file in files:
        request = service.files().get_media(fileId=file['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        pdf_reader = PyPDF2.PdfReader(fh)
        
        total_pages = len(pdf_reader.pages)
        
        # Determine which pages to extract
        if total_pages <= max_pages + last_pages:
            # Document is short enough - extract all pages
            pages_to_extract = list(range(total_pages))
            extraction_note = f"all {total_pages} pages"
        else:
            # Extract first max_pages and last last_pages
            first_pages = list(range(max_pages))
            last_page_indices = list(range(total_pages - last_pages, total_pages))
            pages_to_extract = first_pages + last_page_indices
            extraction_note = f"first {max_pages} + last {last_pages} pages (out of {total_pages})"
        
        # Extract text from selected pages
        text = ""
        for i in pages_to_extract:
            text += pdf_reader.pages[i].extract_text()
        
        documents.append({
            'name': file['name'],
            'text': text,
            'file_id': file['id'],
            'total_pages': total_pages,
            'extracted_pages': len(pages_to_extract)
        })
        
        print(f"Downloaded: {file['name']} ({extraction_note}, {len(text)} characters)")
    
    return documents

def generate_summaries(documents, client, delay=1.0):
    summaries = []
    
    for i, doc in enumerate(documents):
        text = doc['text']
        
        print(f"\n[{i+1}/{len(documents)}] Summarizing: {doc['name']}")
        print(f"  Text length: {len(text)} characters")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""
                        You are a legal expert. Analyze this court case opinion and provide:
                        1. Summary (2–3 sentences). If the case involves artificial intelligence (AI), machine learning (ML), or automated systems, explicitly highlight how the AI/automation is involved in the facts, claims, or holding.
                        2. Key Legal Issue
                        3. Court's Holding
                        4. ELI5 explanation

                        Opinion text:
                        {text}
                        """
                }],
                max_tokens=500
            )
            
            summary = response.choices[0].message.content
            summaries.append(summary)
            print(f"  ✓ Summary generated ({len(summary)} characters)")
            
            # Rate limiting: wait between requests
            if i < len(documents) - 1:
                time.sleep(delay)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            summaries.append(f"Error: {str(e)}")
            # Wait longer if there's an error
            time.sleep(delay * 2)
    
    return summaries

def save_to_json(documents, summaries, output_file):
    """Save documents and summaries to JSON file"""
    data = []
    for doc, summary in zip(documents, summaries):
        data.append({
            'name': doc['name'],
            'file_id': doc['file_id'],
            'full_text': doc['text'],
            'summary': summary,
            'text_length': len(doc['text']),
            'total_pages': doc['total_pages'],
            'extracted_pages': doc['extracted_pages']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_chars = sum(len(d['text']) for d in documents)
    avg_chars = total_chars / len(documents) if documents else 0
    total_pages = sum(d['total_pages'] for d in documents)
    extracted_pages = sum(d['extracted_pages'] for d in documents)
    
    print(f"\nStatistics:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total pages in PDFs: {total_pages}")
    print(f"  Total pages extracted: {extracted_pages}")
    print(f"  Total characters extracted: {total_chars:,}")
    print(f"  Average document length: {avg_chars:,.0f} characters")
    print(f"  Average pages per document: {total_pages/len(documents):.1f}")
    print(f"  Average extracted pages: {extracted_pages/len(documents):.1f}")

def main():
    # Authenticate with Google Drive
    print("\n[1/4] Authenticating with Google Drive...")
    service = authenticate_google_drive()
    
    # Find the folder
    print(f"\n[2/4] Finding folder '{DRIVE_FOLDER_NAME}'...")
    folder_id = find_folder_id(service, DRIVE_FOLDER_NAME)
    if not folder_id:
        print("Exiting...")
        return
    
    # Download PDFs
    print("\n[3/4] Downloading PDFs and extracting text...")
    print(f"  Extracting first {MAX_PAGES} pages + last {LAST_PAGES} pages from each document")
    documents = download_pdfs_from_drive(service, folder_id, max_pages=MAX_PAGES, last_pages=LAST_PAGES)
    
    if not documents:
        print("No documents found. Exiting...")
        return
    
    # Generate summaries
    print("\n[4/4] Generating summaries with OpenAI GPT-4...")
    
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)
    
    summaries = generate_summaries(documents, client, delay=1.0)
    
    # Save to JSON
    print("\n" + "=" * 60)
    print("Saving results...")
    save_to_json(documents, summaries, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("✓ STEP 1 COMPLETE!")
    print("=" * 60)
    print(f"\nNext step: Run 'step2_analyze_and_visualize.py' to generate the visualization")
    print(f"Input file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()