import cv2
from ultralytics import YOLO

YOLO_MODEL_PATH = "yolo_vote_detection_model.pt"
VOTE_SYMBOLS = ['cross' ,'1', '2', '3']
POSITION_TOLERANCE = 20

class YOLOProcessor:
    def __init__(self, model_path):
        # Load a pretrained YOLOv8n model
        self.model = YOLO(model_path)

    def run_inference(self, source_image):
        """Run inference on the source image and return the results."""
        return self.model(source_image)

class VoteProcessor:
    def __init__(self, result):
        self.result = result
        self.class_dict = self.result.names
        self.vote_symbol_dict = self._extract_vote_symbols()

    def _extract_vote_symbols(self):
        """Extracts the positions of symbols that are not the 'name' class."""
        return {self.class_dict[int(box.cls.item())]: box.xyxy.cpu().numpy()
                for box in self.result.boxes
                if self.class_dict[int(box.cls.item())] in VOTE_SYMBOLS}


    def _check_for_vote(self, name_pos, index):
        """Check for the closest vote symbol to the name position."""
        for symbol, symbol_pos in self.vote_symbol_dict.items():
            name_y1 = name_pos[0][1]
            name_y2 = name_pos[0][3]

            # Case 1: Name starts within the symbol bounds
            if name_y1 >= symbol_pos[0][1] and name_y1 <= symbol_pos[0][3]:
                return (index, symbol)
            
            # Case 2: Name fully overlaps the symbol
            elif name_y1 <= symbol_pos[0][1] and name_y2 >= symbol_pos[0][3]:
                return (index, symbol)

            # Case 3: Vote symbol at top of the name
            elif abs(name_y1 - symbol_pos[0][3]) < POSITION_TOLERANCE:
                return (index, symbol)

            # Case 4: Vote symbol at bottom of the name
            elif abs(symbol_pos[0][1] - name_y2) < POSITION_TOLERANCE:
                return (index, symbol)
        return None
    
    def find_names_and_votes(self):
        """Finds the relationship between name positions and vote symbols."""
        names_and_votes = []

        for i, box in enumerate(self.result.boxes):
            class_name = self.class_dict[int(box.cls.item())]
            if class_name not in VOTE_SYMBOLS:
                name_position = box.xyxy.cpu().numpy()
                associated_vote = self._check_for_vote(name_position, i)
                print(i, class_name, associated_vote)
                if associated_vote:
                    names_and_votes.append(associated_vote)

        return names_and_votes

def main():
    # Initialize YOLO processor and read the image
    yolo_processor = YOLOProcessor(YOLO_MODEL_PATH)

    image_path = r'sample_ballot_papers/vote_5.png'
    source_image = cv2.imread(image_path)

    # Run inference on the image
    result = yolo_processor.run_inference(source_image)[0]

    # Process vote and name positions
    vote_processor = VoteProcessor(result)
    names_and_votes = vote_processor.find_names_and_votes()

    # Output the results
    print("="*25)
    print(names_and_votes)
    print("="*25)

    # Visualize the results
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
