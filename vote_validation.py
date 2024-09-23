from typing import List, Dict, Optional

class VoteValidator:
    """Represents a voting ballot and provides methods to check its validity."""

    def __init__(self, valid_vote_characters, valid_vote_numbers):
        self.valid_vote_characters = valid_vote_characters
        self.valid_vote_numbers = valid_vote_numbers
        self.valid_votes = valid_vote_characters + valid_vote_numbers

    def is_valid(self, votes: List[Dict[str, Optional[str]]]) -> bool:
        """
        Check if the ballot is valid based on voting rules.
        """
        extracted_votes = [item['vote'] for item in votes if item['vote'] is not None]
        
        # 1. Check if ballot is empty
        if not extracted_votes:
            return False
        
        # 2. Check only for valid characters
        if any(vote not in self.valid_votes for vote in extracted_votes):
            return False
        
        x_count = sum(extracted_votes.count(char) for char in self.valid_vote_characters)
        
        # 3. If 'X' there should not be any characters
        if x_count > 1 or (x_count == 1 and len(extracted_votes) != 1):
            return False
        
        # 4. Check for valid numbered voting
        if x_count == 0:
            if all(vote in self.valid_vote_numbers for vote in extracted_votes):
                if sorted(extracted_votes) != self.valid_vote_numbers:
                    return False
            else:
                return False
        
        return True