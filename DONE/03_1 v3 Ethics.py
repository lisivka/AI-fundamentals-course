def get_risks_and_categories(risks_list):
    """
    Identify and categorize potential risks of using AI in everyday life.

    Parameters:
        risks_list (list): A list of potential risks and categories associated with AI usage.

    Returns:
        list: A list containing categorized risks.
    """
    categories = []

    print("Welcome to the AI Risks Identification Tool!")
    print("Enter the category of each risk:")

    for risk in risks_list:
        category = input(f"Category for '{risk}': ")
        categories.append((risk, category))

    return categories


# @test_identify_risks
def identify_risks(risks_categories):
    """
    Identify and categorize potential risks of using AI in everyday life.

    Parameters:
        risks_categories (list): A list of potential risks and categories associated with AI usage.

    Returns:
        dict: A dictionary containing categorized risks.
    """
    categorized_risks = {}

    for risk, category in risks_categories:
        if category not in categorized_risks:
            categorized_risks[category] = []
        categorized_risks[category].append(risk)

    return categorized_risks


def display_risks_summary(risks):
    """
    Display a summary of identified risks under each category.

    Parameters:
        risks (dict): A dictionary containing categorized risks.
    """
    print("\nAI Risks Summary:\n")

    for category, category_risks in risks.items():
        print(f"Category: {category}")
        for risk in category_risks:
            print(f"- {risk}")


risks_list = [
    "Privacy concerns: AI-powered devices may collect personal data, raising concerns about how this data is stored and used.",
    "Security vulnerabilities: AI systems could be susceptible to cyberattacks, leading to unauthorized access to personal information.",
    "Bias in AI algorithms: AI algorithms might inadvertently perpetuate biases present in the data they are trained on, leading to unfair or discriminatory outcomes.",
    # Add more AI Risks to this list
]

risks_categories = get_risks_and_categories(risks_list)
risks_data = identify_risks(risks_categories)
display_risks_summary(risks_data)
