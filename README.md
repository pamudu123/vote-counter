# Automated Election Vote Counting


Detailed Explanation of Vote Counting Methods
[Medium Article](https://medium.com/@pamudu1111/automated-election-vote-counting-7b89900f7333)

### 1. Using ChatGPT Vision API to Process Ballot Papers
![Vision API](resources\LLM_proocessing.png)

OpenAI Vision API analyzes images of ballot papers, interprets the votes, and returns structured data.


### 2. Process the Layout with OCR
<table>
  <tr>
    <td>
      <img src="resources/LayoutParse.png" alt="Layout Processing" width="300"/>
    </td>
    <td>
      <img src="resources/LayoutParseLLM.png" alt="Layout Processing LLM" width="300"/>
    </td>
  </tr>
</table>

Layout extraction technique followed by a rule-based/LLM approach to process and extract the details from the structured OCR text.

### 3. Using Object Detection — YOLO
![YOLO vote detection](resources\Yolo_vote_extraction.png)

YOLO object detection model to extract details from the ballot paper.


### 4. Vote Extraction Using Computer Vision
![CV vote detection](resources\CV_method.png)

Pure computer vision-based method, using edge detection, contour detection, kernel processing, and template matching.


-- UNDER CONSTRUCTION ---

