########################## app.py ####################################

import os
import time
import requests
import arxiv
from bs4 import BeautifulSoup
import torch
import streamlit as st
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MAX_INPUT_LENGTH = 1024  # GPT-2's maximum context size
MAX_SEARCH_RESULTS = 2
MAX_BODY_LENGTH = 2000

# --- Force CPU usage ---
device = torch.device('cpu')
print(f"[INFO] Using device: {device}")

# --- Model Loading with Error Handling ---
try:
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    print("[SUCCESS] Model and tokenizer loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    raise SystemExit("Critical error - cannot continue without model")

# --- Enhanced Search Functions ---
def google_search(query: str, num_results: int = MAX_SEARCH_RESULTS) -> list:
    """Safe Google search with error handling and input validation"""
    search_results = []
    
    try:
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all('div', class_='tF2Cxc')[:num_results]

        for g in results:
            try:
                title = g.find('h3').text if g.find('h3') else "No title"
                link = g.find('a')['href'] if g.find('a') else "#"
                snippet = g.find('span', class_='aCOpRe').text if g.find('span', class_='aCOpRe') else "No snippet"

                # Safe content fetching
                body = ""
                try:
                    page = requests.get(link, timeout=5)
                    page_soup = BeautifulSoup(page.text, "html.parser")
                    text = page_soup.get_text(separator=" ", strip=True)
                    body = ' '.join(text.split()[:MAX_BODY_LENGTH])
                except Exception as e:
                    print(f"[WARN] Content fetch failed: {str(e)}")

                search_results.append({
                    "title": title[:200],  # Limit title length
                    "link": link,
                    "snippet": snippet[:500],  # Limit snippet length
                    "body": body[:MAX_BODY_LENGTH]
                })
                time.sleep(1)  # Rate limiting

            except Exception as result_error:
                print(f"[WARN] Failed to process search result: {str(result_error)}")

    except Exception as search_error:
        print(f"[ERROR] Google search failed: {str(search_error)}")
    
    return search_results

def arxiv_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list:
    """Safe Arxiv search with error handling"""
    results = []
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for paper in client.results(search):
            results.append({
                "title": paper.title[:300],  # Limit title length
                "authors": [author.name[:50] for author in paper.authors][:5],  # Limit authors
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "abstract": paper.summary[:1000] if paper.summary else "No abstract",
                "pdf_url": paper.pdf_url if paper.pdf_url else "#"
            })
            time.sleep(1)  # Rate limiting
            
    except Exception as e:
        print(f"[ERROR] Arxiv search failed: {str(e)}")
    
    return results

# --- Robust Agents with Input Validation ---
class GoogleSearchAgent:
    def run(self, topic):
        print(f"[Google Search Agent] Searching for: {topic[:50]}...")  # Truncate long queries
        safe_topic = topic[:100]  # Limit query length
        return google_search(safe_topic), f"Google search for: {safe_topic}"

class ArxivSearchAgent:
    def run(self, topic):
        print(f"[Arxiv Search Agent] Searching for: {topic[:50]}...")
        safe_topic = topic[:100]  # Limit query length
        return arxiv_search(safe_topic), f"Arxiv search for: {safe_topic}"

class ReportAgent:
    def run(self, topic, google_results, arxiv_results):
        print("[Report Agent] Generating report...")
        
        # Safe prompt construction
        truncated_topic = topic[:200]
        safe_google = str(google_results)[:1500]  # Limit context size
        safe_arxiv = str(arxiv_results)[:1500]
        
        prompt = f"""Write a literature review about: {truncated_topic}
        Google Results: {safe_google}
        Arxiv Papers: {safe_arxiv}
        Requirements: Formal tone, synthesize key findings, reference Hayek at the end.
        """
        
        try:
            # Safe tokenization with truncation
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            ).to(device)

            # Safe generation with output checks
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=min(1500, MAX_INPUT_LENGTH),
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )

            if outputs.shape[1] > 0:
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                text = text[:5000]  # Limit output length
            else:
                text = "Failed to generate review - empty output"
                
        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            text = "Error generating literature review"
        
        return text, prompt

# --- Team Implementation with Fallbacks ---
class LiteratureReviewTeam:
    def __init__(self):
        self.google_agent = GoogleSearchAgent()
        self.arxiv_agent = ArxivSearchAgent()
        self.report_agent = ReportAgent()

    def conduct_review(self, topic):
        try:
            google_results, google_prompt = self.google_agent.run(topic)
            arxiv_results, arxiv_prompt = self.arxiv_agent.run(topic)
            
            # Validate results before passing to report agent
            if not isinstance(google_results, list):
                google_results = []
            if not isinstance(arxiv_results, list):
                arxiv_results = []
                
            report, report_prompt = self.report_agent.run(
                topic[:200],  # Truncate long topics
                google_results[:MAX_SEARCH_RESULTS],  # Enforce max results
                arxiv_results[:MAX_SEARCH_RESULTS]
            )
            
            return {
                'google_prompt': google_prompt,
                'arxiv_prompt': arxiv_prompt,
                'report_prompt': report_prompt,
                'google_results': google_results,
                'arxiv_results': arxiv_results,
                'literature_review': report
            }
            
        except Exception as e:
            print(f"[CRITICAL] Team error: {str(e)}")
            return {
                'error': f"System error: {str(e)}",
                'literature_review': "Failed to generate review due to system error"
            }

# --- Streamlit UI with Safe Rendering ---
def main():
    st.title("📚 Robust Literature Review Generator")
    st.write("Safe search and AI-powered review generation")
    
    topic = st.text_input("Research Topic:", max_chars=200)
    
    if st.button("Generate Review"):
        if not topic.strip():
            st.error("Please enter a valid topic!")
            return
            
        with st.spinner("Conducting safe review process..."):
            team = LiteratureReviewTeam()
            result = team.conduct_review(topic)
            
            if 'error' in result:
                st.error(result['error'])
                return
                
            st.success("Review process completed!")
            
            # Safe results display
            st.subheader("Search Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Google Results**")
                if result['google_results']:
                    for i, res in enumerate(result['google_results'][:MAX_SEARCH_RESULTS]):
                        st.write(f"### Result {i+1}")
                        st.write(f"**Title**: {res.get('title', 'N/A')}")
                        st.write(f"**Preview**: {res.get('snippet', 'No snippet')[:300]}...")
                        st.write(f"[Link]({res.get('link', '#')})")
                else:
                    st.write("No Google results found")
                    
            with col2:
                st.write("**Arxiv Papers**")
                if result['arxiv_results']:
                    for i, paper in enumerate(result['arxiv_results'][:MAX_SEARCH_RESULTS]):
                        st.write(f"### Paper {i+1}")
                        st.write(f"**Title**: {paper.get('title', 'N/A')}")
                        st.write(f"**Authors**: {', '.join(paper.get('authors', ['Unknown']))[:100]}...")
                        st.write(f"[PDF]({paper.get('pdf_url', '#')})")
                else:
                    st.write("No Arxiv papers found")
            
            # Review display with safety checks
            st.subheader("Generated Literature Review")
            review_text = result.get('literature_review', 'No review generated')
            st.write(review_text[:10000])  # Limit display length
            
            # Show technical details in expander
            with st.expander("Technical Details"):
                st.write("**Google Prompt:**", result.get('google_prompt', 'N/A'))
                st.write("**Arxiv Prompt:**", result.get('arxiv_prompt', 'N/A'))
                st.write("**Report Prompt:**", result.get('report_prompt', 'N/A')[:2000])

if __name__ == "__main__":
    main()
