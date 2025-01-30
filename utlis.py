from config import PROMPT

def update_prompt(age, gender):

    # Map the gender selection to male/female
    gender_map = {
        "Man": "male",
        "Woman": "female"
    }
    
    # Replace the placeholders in the prompt
    updated_prompt = PROMPT.replace("[Age]", str(age))
    updated_prompt = updated_prompt.replace("[Gender]", gender_map[gender])
    
    return updated_prompt