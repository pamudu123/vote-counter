import base64
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

################ LLM OUTPUT SCHEMA ################ 
class Candidate(BaseModel):
    """Represents a candidate on the ballot."""
    name: str = Field(..., description="Name of the candidate")
    position: int = Field(..., description="Position of the candidate on the ballot sheet")

class BallotPaper(BaseModel):
    """Represents the result of analyzing a ballot."""
    is_valid: bool = Field(..., description="Indicates whether the ballot is valid")
    validity_explanation: str = Field(..., description="Explanation for why the ballot is valid or invalid")
    uses_cross_or_numbering: bool = Field(..., description="True if the ballot contains a cross or numbering, False otherwise")
    first_vote: Optional[Candidate] = Field(None, description="Details of the candidate marked as the first preference")
    second_vote: Optional[Candidate] = Field(None, description="Details of the candidate marked as the second preference")
    third_vote: Optional[Candidate] = Field(None, description="Details of the candidate marked as the third preference")

################ LLM PROCESSING ################
class BallotAnalyzer:
    """Handles the analysis of ballot images using GPT."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = instructor.patch(OpenAI())
        self.model = model

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def _get_analysis_prompt() -> str:
        return """
        Analyze the provided ballot paper image to determine the validity of the vote and extract the preference order.

        Steps:
        1. Image Analysis:
            - Scan the ballot paper for the presence of numbers (1, 2, 3) or a cross sign (X) next to candidates' names.

        2. Determine Validity:
            - Valid Ballot: 
                - Contains only a cross sign (X) without any numbers.
                - Contains numbers (1, 2, 3) without a cross sign.
            - Invalid Ballot:
                - Contains both a cross sign (X) and numbers (1, 2, 3).
                - Contains incorrect or duplicated numbering, or is missing required numbers.

        3. Extract Vote Preferences:
            - If the ballot is valid, extract the candidate names and their positions on the ballot sheet corresponding to the votes.
            - If the cross sign is present, treat the candidate marked with the cross as the first preference.
            - Ensure no other markings or errors are present that would invalidate the ballot.

        4. Output the Following JSON Structure:

        {
        "validity": true or false,
        "explanation_for_validity": "Explanation for why the ballot is valid or invalid. Add the extracted details on the ballot paper",
        "cross_or_numbering": true or false,
        "votes": {
            "1st_vote": {
                "name": "Candidate Name",
                "position_on_ballot_sheet": "Position Number"
            },
            "2nd_vote": {
                "name": "Candidate Name",
                "position_on_ballot_sheet": "Position Number"
            },
            "3rd_vote": {
                "name": "Candidate Name",
                "position_on_ballot_sheet": "Position Number"
            }
        }
        }

        Example Cases:
            1. Valid Ballot with Cross Only:
            - Image Analysis: Detect a cross sign next to a candidate.
            - Validity: True
            - Explanation: "The ballot is valid with only a cross sign present, no numbers."
            - Cross or Numbering: True
            - Votes: 
                - 1st Vote: {"name": "Mahesha Hettiarachchi", "position_on_ballot_sheet": "6"}

        2. Valid Ballot with Numbering Only:
            - Image Analysis: Detect numbers 1, 2, 3 next to candidates.
            - Validity: True
            - Explanation: "The ballot is valid with numbers 1, 2, 3 clearly marked."
            - Cross or Numbering: False
            - Votes: 
                - 1st Vote: {"name": "Tharindu Fernando", "position_on_ballot_sheet": "3"}
                - 2nd Vote: {"name": "Shenal Rathnayake", "position_on_ballot_sheet": "4"}
                - 3rd Vote: {"name": "Mahesha Hettiarachchi", "position_on_ballot_sheet": "6"}

        3. Invalid Ballot with Cross and Numbering Both Present:
            - Image Analysis: Detect both cross sign and numbers.
            - Validity: False
            - Explanation: "The ballot is invalid because it contains both a cross sign and numbers."
            - Cross or Numbering: False
            - Votes: {}
        """

    def analyze_ballot(self, image_path: str) -> BallotPaper:
        """Analyze the ballot image and return the result."""
        base64_image = self._encode_image(image_path)
        
        return self.client.chat.completions.create(
                model = self.model,
                response_model = BallotPaper,
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self._get_analysis_prompt()}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}],
                    }
                ],
                max_tokens = 1000,
                )

def main():
    image_path = "sample_ballot_papers/vote_1.png"
    analyzer = BallotAnalyzer()

    # Analyze ballot
    analysis_result = analyzer.analyze_ballot(image_path)
    print(analysis_result)

    # Print analysis results
    print(f"Validity: {analysis_result.is_valid}")
    print(f"Explanation: {analysis_result.validity_explanation}")
    print(f"Uses Cross or Numbering: {analysis_result.uses_cross_or_numbering}")
    print(f"First Preference: {analysis_result.first_vote}")
    print(f"Second Preference: {analysis_result.second_vote}")
    print(f"Third Preference: {analysis_result.third_vote}")


if __name__ == "__main__":
    main()