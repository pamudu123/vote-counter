import os
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from unstract.llmwhisperer.client import LLMWhispererClient, LLMWhispererClientException

from vote_validation import VoteValidator

# Constants
VALID_VOTE_CHARACTERS = ['X', 'x']
VALID_VOTE_NUMBERS = ['1', '2', '3']
VALID_VOTES = VALID_VOTE_CHARACTERS + VALID_VOTE_NUMBERS
CANDIDATE_NAMES = [
    'PAMUDU RANASINGHE', 'KASUN JAYAWARDENA', 'THARINDU FERNANDO',
    'SHENAL RATHNAYAKE', 'MAHESHA HETTIARACHCHI', 'RAVINDU WICKRAMASINGHE',
    'MANUJA WIJESINGHE', 'ISURU KARUNARATNE'
]
NAME_END_POSITION = 20  # Assumed position where candidate name ends in the ballot line

class LayoutProcessor:
    """Handles the conversion of PDF documents to structured text."""
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('LLMWHISPER_API_KEY')
        base_url = os.getenv('LLMWHISPER_BASE_URL')
        self.client = LLMWhispererClient(base_url=base_url, api_key=api_key, logging_level="INFO")

    def img_to_structured_text(self, pdf_path: str) -> str:
        """Convert a PDF file to structured text using LLMWhisperer."""
        try:
            result = self.client.whisper(
                file_path=pdf_path,
                processing_mode='ocr',
                output_mode='line-printer',
                force_text_processing=False,
                line_splitter_tolerance=0.4,
                horizontal_stretch_factor=1.2
            )
            return result["extracted_text"]
        except LLMWhispererClientException as e:
            return f'PDF conversion failed with error: {e}'

class VoteExtractor:
    """Extract votes and candidate names from text."""
    @staticmethod
    def extract_vote(line: str) -> Optional[str]:
        """Extract a valid vote from a line of text."""
        line = line.replace('[X]', '')
        for char in line:
            if char in VALID_VOTES:
                return char
        return None

    @staticmethod
    def extract_candidate_name(text: str) -> Optional[str]:
        """Extract a candidate name from a line of text."""
        processed_names = [''.join(name.split()).upper() for name in CANDIDATE_NAMES]
        processed_input = ''.join(text.split()).upper()
        for i, processed_name in enumerate(processed_names):
            if processed_name in processed_input:
                return CANDIDATE_NAMES[i]
        return None

class VotingSystem:
    """Orchestrates the entire voting process, from PDF processing to vote extraction."""
    def __init__(self, pdf_path: str):
        self.pdf_processor = LayoutProcessor()
        self.pdf_path = pdf_path

    def process_votes(self) -> List[Dict[str, Optional[str]]]:
        """Process the votes from the PDF ballot."""
        extracted_text = self.pdf_processor.img_to_structured_text(self.pdf_path)
        lines = [line.strip() for line in extracted_text.split('\n')]
        
        vote_dict = []
        candidate_list_index = 0

        for i, line in enumerate(lines):
            candidate_name = VoteExtractor.extract_candidate_name(line)
            if candidate_name:
                # Try to extract vote from current line, then previous, then next
                vote = VoteExtractor.extract_vote(line[NAME_END_POSITION:])
                if vote is None:
                    vote = VoteExtractor.extract_vote(lines[i-1][NAME_END_POSITION:]) or \
                           VoteExtractor.extract_vote(lines[i+1][NAME_END_POSITION:])
                
                candidate_list_index += 1
                record = {
                    'sheet_position': candidate_list_index,
                    'candidate_name': candidate_name,
                    'vote': vote
                }
                vote_dict.append(record)

        return vote_dict

def main():
    voting_system = VotingSystem('sample_ballot_papers/vote_1.png')
    votes = voting_system.process_votes()
    validator = VoteValidator(VALID_VOTE_CHARACTERS, VALID_VOTE_NUMBERS)
    
    print("Extracted votes:")
    for vote in votes:
        print(vote)
    
    print(f"\nBallot is valid: {validator.is_valid(votes)}")

if __name__ == "__main__":
    main()