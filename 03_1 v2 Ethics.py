risks_list = [
    "Privacy concerns: AI-powered devices may collect personal data, raising concerns about how this data is stored and used.",
    "Security vulnerabilities: AI systems could be susceptible to cyberattacks, leading to unauthorized access to personal information.",
    "Bias in AI algorithms: AI algorithms might inadvertently perpetuate biases present in the data they are trained on, leading to unfair or discriminatory outcomes.",
    # Add more AI Risks if needed
]

def get_risks_and_categories(risks_list):
    """
    Identify and categorize potential risks of using AI in everyday life.

    Parameters:
        risks_list (list): A list of potential risks and categories associated with AI usage.

    Returns:
        list: A list containing categorized risks.
    """

    print("Welcome to the AI Risks Identification Tool!")
    print("Enter the category of each risk:")

    # Initialize an empty list to store categorized risks
    risks_categories = []

    for risk_description in risks_list:
        print(f"Risk Description: {risk_description}")
        category = input("Enter the category: ")

        # Append the category and risk description as a list to the risks_categories list
        risks_categories.append([category, risk_description])

    # Return the list of categorized risks
    return risks_categories



def identify_risks(risks_categories):
    """
    Identify and categorize potential risks of using AI in everyday life.

    Parameters:
        risks_categories (list): A list of potential risks and categories associated with AI usage.

    Returns:
        dict: A dictionary containing categorized risks.
    """
    categorized_risks = {}

    for category, risk_description in risks_categories:
        if category not in categorized_risks:
            categorized_risks[category] = []

        categorized_risks[category].append(risk_description)

    return categorized_risks




def display_risks_summary(risks):
    """
    Display a summary of identified risks under each category.

    Parameters:
        risks (dict): A dictionary containing categorized risks.
    """
    print("\nAI Risks Summary:\n")

    for category, risk_descriptions in risks.items():
        print(f"Category: {category}")
        for risk_description in risk_descriptions:
            print(f"- {risk_description}")


# Call the function with the categorized risks data




# Sample data

# Call the function with the categorized risks data obtained from the get_risks_and_categories function


risks_categories = get_risks_and_categories(risks_list)
risks_data = identify_risks(risks_categories)
print(f"risks_categories: {risks_categories}")
risks_data = identify_risks(risks_categories)
display_risks_summary(risks_data)

display_risks_summary(risks_data)