import re

def extract_key_requirements(text: str) -> list:
    """
    Extract bullet-point requirements or lines starting with '-','*'.
    """
    lines = text.split('\n')
    bullet_points = []
    for line in lines:
        # If the line begins with a dash or asterisk (typical bullet points)
        if line.strip().startswith('-') or line.strip().startswith('*'):
            bullet_points.append(line.strip())
    
    # If no bullet points exist, try simple keyword scanning
    if not bullet_points:
        possible_requirements = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ["requirement", "must have", "qualifications"]):
                possible_requirements.append(line.strip())
        bullet_points = possible_requirements
    
    return bullet_points

def extract_company_values(text: str) -> list:
    """
    Extract lines that contain keywords like 'mission', 'vision', 'value'.
    """
    lines = text.split('\n')
    values = []
    # Common keywords that might appear in a company's mission or values statements
    keywords = ["mission", "vision", "value", "core values", "principle", "culture"]
    for line in lines:
        # If line has any of the keywords, store it
        if any(keyword in line.lower() for keyword in keywords):
            values.append(line.strip())
    
    return values
